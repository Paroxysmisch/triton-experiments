import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    out_logexpsum,  # [batch, head]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_out_logexpsum_b, stride_out_logexpsum_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    sum_exp = 0.0
    max_logic = float("-1e20")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n * stride_mid_o_es)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    if block_n_size > 0:
        # Here we check whether block_n_size is 0 in order to avoid "div by zero" error
        tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
        tl.store(out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h, max_logic + tl.log(sum_exp))
    return

import torch
from triton import cdiv

def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, out_logexpsum, BLOCK_SEQ, BLOCK_DMODEL):
    assert B_Seqlen.dim() == 1
    assert Mid_O.dim() == 4
    assert Mid_O_LogExpSum.dim() == 3
    assert O.dim() == 3
    assert out_logexpsum.dim() == 2

    B, H, S, D = Mid_O.shape
    assert B == B_Seqlen.shape[0]
    assert H == Mid_O_LogExpSum.shape[1]
    assert S == Mid_O_LogExpSum.shape[2]
    assert D == BLOCK_DMODEL
    assert O.shape == (B, H, D)
    assert out_logexpsum.shape == (B, H)

    grid = (B, H)

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,  # [batch, head, seq_block_num, head_dim]
        Mid_O_LogExpSum,  # [batch, head, seq_block_num]
        O,  # [batch, head, head_dim]
        out_logexpsum,  # [batch, head]
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1), Mid_O_LogExpSum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        out_logexpsum.stride(0), out_logexpsum.stride(1),
        BLOCK_SEQ, BLOCK_DMODEL
    )
    return O, out_logexpsum
