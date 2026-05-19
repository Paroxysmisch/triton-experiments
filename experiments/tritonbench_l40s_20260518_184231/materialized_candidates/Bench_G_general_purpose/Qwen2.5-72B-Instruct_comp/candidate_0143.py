import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_om, stride_on,
    batch, chunk, group, head,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    IS_CAUSAL: tl.constexpr, 
    HAS_SEQ_IDX: tl.constexpr,
    seq_idx_ptr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_mn = pid % num_pid_in_group
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        acc += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if IS_CAUSAL:
        mask = offs_m[:, None] >= offs_n[None, :]
        acc = tl.where(mask, acc, 0.0)

    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + group_id)
        mask = seq_idx == group_id
        acc = tl.where(mask, acc, 0.0)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc)

import torch
import triton
import triton.language as tl

def _bmm_chunk_fwd(a, b, batch, chunk, group, head, M, N, K, IS_CAUSAL=False, HAS_SEQ_IDX=False, seq_idx=None):
    # Ensure input tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()

    # Allocate output tensor
    out = torch.empty((batch, chunk, group, head, M, N), dtype=a.dtype, device=a.device)

    # Compute grid size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    num_groups = batch * chunk * group * head
    grid = (num_groups * num_pid_in_group,)

    # Launch kernel
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(4), a.stride(5),
        b.stride(4), b.stride(5),
        out.stride(4), out.stride(5),
        batch, chunk, group, head,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        IS_CAUSAL, HAS_SEQ_IDX,
        seq_idx
    )

    return out
