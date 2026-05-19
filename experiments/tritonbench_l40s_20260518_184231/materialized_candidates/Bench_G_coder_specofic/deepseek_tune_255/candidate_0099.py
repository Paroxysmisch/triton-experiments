import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Out,
    stride_b_seqlen_b,
    stride_b_seqlen_s,
    stride_mid_o_b,
    stride_mid_o_h,
    stride_mid_o_s,
    stride_mid_o_d,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    head_num: tl.constexpr,
    block_seq: tl.constexpr,
    block_dmodel: tl.constexpr,
    sm_scale: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_head_o = cur_head * block_dmodel

    sum_exp = 0.0
    max_logic = float("-inf")
    acc = tl.zeros([block_seq, block_dmodel], dtype=tl.float32)

    cur_seq_len = tl.load(B_Seqlen + cur_batch * stride_b_seqlen_b)
    block_n_size = (cur_seq_len + block_seq - 1) // block_seq

    offs_s = tl.arange(0, block_seq)
    cur_batch_seq_len = cur_seq_len

    for block_n in range(0, block_n_size):
        beg = block_n * block_seq
        end = tl.minimum((block_n + 1) * block_seq, cur_batch_seq_len)
        beg_o = beg * block_dmodel
        offs_d = tl.arange(0, block_dmodel)
        offs_m = cur_batch * stride_mid_o_b + cur_head * stride_mid_o_h + (beg_o + offs_d)
        tlogic = tl.load(Mid_O_LogExpSum + offs_m[None, :])
        tval = tl.load(Mid_O + offs_m[None, :])
        max_logic = tl.maximum(max_logic, tl.max(tlogic, 1))
        logic_scale = tl.exp((tlogic - max_logic) * sm_scale)
        acc *= tl.exp((sum_exp - max_logic) * sm_scale)
        acc += tl.dot(tval, logic_scale.to(tval.dtype), allow_tf32=False)
        sum_exp = tl.maximum(sum_exp, tl.sum(tlogic, 1).to(sum_exp.dtype))

    offs_d = tl.arange(0, block_dmodel)
    offs_m = cur_batch * stride_out_b + cur_head * stride_out_h + (cur_head_o + offs_d)
    tl.store(Out + offs_m[None, :], acc.to(Out.dtype))


def flash_decode_stage2(
    B_Seqlen: Tensor,
    Mid_O: Tensor,
    Mid_O_LogExpSum: Tensor,
    Out: Tensor,
    max_len_in_batch: int,
    head_num: int,
    sm_scale: float,
):
    seq_len_dim = B_Seqlen.ndim - 1
    mid_out_dim = Mid_O.ndim - 2

    assert all([d > 0 for d in B_Seqlen.shape[:seq_len_dim]])
    assert all([d > 0 for d in Mid_O.shape[:mid_out_dim]])
    assert B_Seqlen.shape[0] == Mid_O.shape[0] == Out.shape[0]
    assert all([B_Seqlen.shape[0] == mid_dim for mid_dim in [Mid_O.shape[i] for i in range(mid_out_dim)]])
    assert all([mid_o_dim == out_dim for mid_o_dim, out_dim in zip(Mid_O.shape[-1:0:-1], Out.shape[-1:0:-1])])
    assert mid_out_dim == Out.ndim - 2
    assert Mid_O_LogExpSum.shape == Mid_O.shape
    assert B_Seqlen.is_contiguous()
    assert Mid_O.is_contiguous()
    assert Mid_O_LogExpSum.is_contiguous()
    assert Out.is_contiguous()
    assert B_Seqlen.size(0) == Out.size(0)
    assert B_Seqlen.size(0) == Mid_O.size(0) == Mid_O_LogExpSum.size(0)
    assert all([b_seq_len <= max_len_in_batch for b_seq_len in B_Seqlen])

    batch = B_Seqlen.size(0)
    block_seq = max_len_in_batch // 2
    block_dmodel = Mid_O.shape[-1]
    grid = (batch, head_num)

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        B_Seqlen.stride(0),
        B_Seqlen.stride(1),
        Mid_O.stride(0),
        Mid_O.stride(1),
        Mid_O.stride(2),
        Mid_O.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        head_num,
        block_seq,
        block_dmodel,
        sm_scale,
        num_warps=8,
        num_stages=2,
    )
