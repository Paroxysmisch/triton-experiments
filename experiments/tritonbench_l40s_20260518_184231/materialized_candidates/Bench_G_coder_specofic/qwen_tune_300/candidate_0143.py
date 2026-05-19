import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a, b, out, seq_idx,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm, stride_outn,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program ids `pid` to the block of C it should compute.
    pid_bat = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    # for sequence parallel
    pid_seq = tl.program_id(axis=3)
    # compute the block of C it should compute.
    # we use `//` and `%` instead of `div_rn` and `mod` because triton compiler
    # is not friendly with tensor argument in div_rn and mod
    c_m = pid_c // chunk_size
    c_n = pid_c % chunk_size
    # batch and head indices
    bat_idx = stride_a_batch * pid_bat + stride_seq_idx_batch * pid_seq
    head_idx = stride_a_head * pid_h
    # The memory address of all the elements that we want to load can be computed as follows
    a_offs = (bat_idx + stride_a_batch * 0 + stride_a_seqlen * c_m + stride_a_head * 0 + tl.arange(0, BLOCK_SIZE_M) * stride_a_seqlen + tl.arange(0, BLOCK_SIZE_K) * stride_ak)[None, :] + (head_idx + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_a_head)
    b_offs = (bat_idx + stride_b_batch * 0 + stride_b_seqlen * c_n + stride_b_head * 0 + tl.arange(0, BLOCK_SIZE_N) * stride_b_seqlen + tl.arange(0, BLOCK_SIZE_K) * stride_bk)[None, :] + (head_idx + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_b_head)
    # Create a mask to guard memory operations against out-of-bounds accesses
    a_mask = (c_m * chunk_size + tl.arange(0, BLOCK_SIZE_M)) < seqlen
    b_mask = (c_n * chunk_size + tl.arange(0, BLOCK_SIZE_N)) < seqlen
    # Load a and b from DRAM, mask out-of-bounds elements
    a = tl.load(a + a_offs, mask=a_mask, other=0.0)
    b = tl.load(b + b_offs, mask=b_mask, other=0.0)
    # do not support int8 now
    # accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    # We don't want to do a reduction over K if K=1, so we add a check for that
    if BLOCK_SIZE_K == 1:
        acc += tl.dot(a, b)
    else:
        for k in range(BLOCK_SIZE_K):
            acc += tl.dot(a[:, k], b[k])
    if HAS_SEQ_IDX:
        seq_idx_offs = (bat_idx + tl.arange(0, BLOCK_SIZE_M) * stride_seq_idx_batch + tl.arange(0, BLOCK_SIZE_N) * stride_seq_idx_seqlen)[None, :] + stride_seq_idx_batch * pid_seq
        seq_idx_mask = (c_m * chunk_size + tl.arange(0, BLOCK_SIZE_M)) < seqlen and (c_n * chunk_size + tl.arange(0, BLOCK_SIZE_N)) < seqlen
        # Load sequence index from DRAM
        seq_idx_ptr = seq_idx + seq_idx_offs
        cur_seq_idx = tl.load(seq_idx_ptr, mask=seq_idx_mask, other=-1)
        # barrier is required to prevent compiler from reordering the seq_idx load
        cur_seq_idx = tl.where(cur_seq_idx != -1, cur_seq_idx, -1)
        # if sequence index does not match, zero out the accumulated chunk
        acc = tl.where(cur_seq_idx == pid_seq, acc, 0.0)
    if IS_CAUSAL:
        causal_mask = (c_m * chunk_size + tl.arange(0, BLOCK_SIZE_M)) >= (c_n * chunk_size + tl.arange(0, BLOCK_SIZE_N))
        acc = tl.where(causal_mask, acc, 0.0)
    # Store acc in DRAM.
    out_offs = (bat_idx + stride_out_batch * 0 + stride_out_chunk * pid_c + stride_out_head * pid_h)[None, :] + (tl.arange(0, BLOCK_SIZE_M) * stride_outm + tl.arange(0, BLOCK_SIZE_N) * stride_outn)
    out_mask = (c_m * chunk_size + tl.arange(0, BLOCK_SIZE_M)) < seqlen and (c_n * chunk_size + tl.arange(0, BLOCK_SIZE_N)) < seqlen
    tl.store(out + out_offs, acc, mask=out_mask)


def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False):
    # shape constraints
    assert a.shape[1] == b.shape[1]
    assert a.shape[2] == b.shape[2]
    assert a.is_contiguous()
    batch, seqlen, K = a.shape
    _, _, ngroups = a.stride()
    c = torch.empty((batch, seqlen, K), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (batch, triton.cdiv(seqlen, META['BLOCK_SIZE_M']) * triton.cdiv(seqlen, META['BLOCK_SIZE_N']), ngroups, 1)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    if K <= 16:
        BLOCK_SIZE_K = 16
    elif K <= 32:
        BLOCK_SIZE_K = 32
    elif K <= 64:
        BLOCK_SIZE_K = 64
    HAS_SEQ_IDX = 0 if seq_idx is None else 1
    # enqueue kernel
    _bmm_chunk_fwd_kernel[grid](
        a, b, c, seq_idx,
        seqlen, chunk_size, K, ngroups,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        seq_idx.stride(0), seq_idx.stride(1) if HAS_SEQ_IDX else 0,
        causal, HAS_SEQ_IDX,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
