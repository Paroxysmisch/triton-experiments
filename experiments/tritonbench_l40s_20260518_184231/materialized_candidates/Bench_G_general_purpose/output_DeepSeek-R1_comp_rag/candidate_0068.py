import torch
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
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d * stride_mid_od
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    
    for block_seq_n in range(block_n_size):
        v_ptrs = Mid_O + offs_v + block_seq_n * stride_mid_os
        tv = tl.load(v_ptrs)
        logic_ptrs = Mid_O_LogExpSum + offs_logic + block_seq_n * stride_mid_o_es
        tlogic = tl.load(logic_ptrs)
        
        new_max_logic = tl.maximum(max_logic, tlogic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc = acc * old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    if block_n_size > 0:
        logexpsum_result = max_logic + tl.log(sum_exp)
        O_ptrs = O + cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
        tl.store(O_ptrs, acc / sum_exp)
        out_ptrs = out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h
        tl.store(out_ptrs, logexpsum_result)

def flash_decode_stage2(
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    B_Seqlen: torch.Tensor,
    O: torch.Tensor,
    out_logexpsum: torch.Tensor,
    BLOCK_SEQ: int = 128
):
    assert Mid_O.is_cuda and Mid_O_LogExpSum.is_cuda and B_Seqlen.is_cuda and O.is_cuda and out_logexpsum.is_cuda
    assert Mid_O.is_contiguous(), "Mid_O must be contiguous"
    assert Mid_O_LogExpSum.is_contiguous(), "Mid_O_LogExpSum must be contiguous"
    assert O.is_contiguous(), "O must be contiguous"
    assert out_logexpsum.is_contiguous(), "out_logexpsum must be contiguous"
    assert B_Seqlen.dtype == torch.int32, "B_Seqlen must be torch.int32"
    
    batch, num_heads, seq_block_num, head_dim = Mid_O.shape
    assert Mid_O_LogExpSum.shape == (batch, num_heads, seq_block_num), "Mid_O_LogExpSum shape mismatch"
    assert O.shape == (batch, num_heads, head_dim), "O shape mismatch"
    assert out_logexpsum.shape == (batch, num_heads), "out_logexpsum shape mismatch"
    assert B_Seqlen.shape == (batch,), "B_Seqlen must be 1D tensor with size equal to batch"
    
    BLOCK_DMODEL = head_dim
    assert (BLOCK_DMODEL & (BLOCK_DMODEL - 1)) == 0, "head_dim must be a power of two"
    
    grid = (batch, num_args := num_heads)
    
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        O,
        out_logexpsum,
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1), Mid_O_LogExpSum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        out_logexpsum.stride(0), out_logexpsum.stride(1),
        BLOCK_SEQ,
        BLOCK_DMODEL
    )
    
    return O, out_logexpsum
