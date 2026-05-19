import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_logics_b, stride_logics_h, stride_logics_d,
    stride_v_b, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    stride_b_loc_b, stride_b_loc_s,
    stride_b_start_loc_b, stride_b_start_loc_s,
    stride_b_seqlen_b,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = tl.load(B_Start_Loc + cur_batch)

    logic_ptrs = Logics + cur_head * stride_logics_h + cur_batch * stride_logics_b
    v_ptrs = V + cur_head * stride_v_h + cur_batch * stride_v_b
    out_ptrs = Out + cur_head * stride_out_h + cur_batch * stride_out_b

    e_max = -float("inf")
    p = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_ptrs_block = v_ptrs + (start_n + offs_n) * stride_v_d + offs_d
        logic_ptrs_block = logic_ptrs + (start_n + offs_n) * stride_logics_d
        v = tl.load(v_ptrs_block, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
        e = tl.load(logic_ptrs_block, mask=(start_n + offs_n) < cur_batch_seq_len, other=-float("inf")).to(tl.float32)
        e_max = tl.maximum(e_max, tl.max(e, 0))
        p += tl.exp(e - e_max)
        acc += tl.sum(v * tl.exp(e - e_max), axis=0)

    p = p / cur_batch_seq_len
    e_max = e_max.to(V.dtype.element_ty)

    acc = acc / p
    acc = acc.to(V.dtype.element_ty)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask = (start_n + offs_n) < cur_batch_seq_len
        v_ptrs_block = v_ptrs + (start_n + offs_n) * stride_v_d + offs_d
        logic_ptrs_block = logic_ptrs + (start_n + offs_n) * stride_logics_d
        out_ptrs_block = out_ptrs + (cur_batch_start_index + start_n + offs_n) * stride_out_d + offs_d
        v = tl.load(v_ptrs_block, mask=mask, other=0.0)
        e = tl.load(logic_ptrs_block, mask=mask, other=-float("inf")).to(tl.float32)
        e = tl.exp(e - e_max)
        p = tl.exp(e) / p
        tl.store(out_ptrs_block, acc * p + v * (1 - p))


def token_softmax_reducev_fwd(logics, v, out, B_Loc, B_Start_Loc, B_Seqlen, num_batches, num_heads, max_seq_len, dim, num_warps, num_stages):
    BLOCK_N = triton.next_power_of_2(max_seq_len)
    grid = (num_batches, num_heads)
    num_sm = 8 * num_warps
    BLOCK_DMODEL = triton.next_power_of_2(dim)
    _fwd_kernel[grid](
        logics, v, out,
        B_Loc, B_Start_Loc, B_Seqlen,
        logics.stride(0), logics.stride(1), logics.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B_Loc.stride(0), B_Loc.stride(1),
        B_Start_Loc.stride(0), B_Start_Loc.stride(1),
        B_Seqlen.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=1,
        num_sm=num_sm,
    )
