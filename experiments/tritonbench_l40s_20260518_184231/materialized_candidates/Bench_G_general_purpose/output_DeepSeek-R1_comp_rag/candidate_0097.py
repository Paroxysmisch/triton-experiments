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
        inv_sum_exp = 1.0 / sum_exp
        acc *= inv_sum_exp
        tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od, acc)
        tl.store(out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h,
                 max_logic + tl.log(sum_exp))

def flash_decode_stage2(
    B_Seqlen: torch.Tensor,
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    BLOCK_SEQ: int = 128
):
    batch, head_num, seq_block_num, head_dim = Mid_O.shape
    device = Mid_O.device
    dtype = Mid_O.dtype
    
    # Initialize output tensors with zeros
    O = torch.zeros((batch, head_num, head_dim), device=device, dtype=dtype)
    out_logexpsum = torch.zeros((batch, head_num), device=device, dtype=dtype)
    
    BLOCK_DMODEL = head_dim  # Ensure head_dim is a power of two for optimal performance
    
    grid = (batch, head_num)
    
    # Compute strides for Mid_O
    stride_mid_ob = Mid_O.stride(0)
    stride_mid_oh = Mid_O.stride(1)
    stride_mid_os = Mid_O.stride(2)
    stride_mid_od = Mid_O.stride(3)
    
    # Compute strides for Mid_O_LogExpSum
    stride_mid_o_eb = Mid_O_LogExpSum.stride(0)
    stride_mid_o_eh = Mid_O_LogExpSum.stride(1)
    stride_mid_o_es = Mid_O_LogExpSum.stride(2)
    
    # Compute strides for O
    stride_obs = O.stride(0)
    stride_oh = O.stride(1)
    stride_od = O.stride(2)
    
    # Compute strides for out_logexpsum
    stride_out_logexpsum_b = out_logexpsum.stride(0)
    stride_out_logexpsum_h = out_logexpsum.stride(1)
    
    # Launch kernel
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        O,
        out_logexpsum,
        stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
        stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
        stride_obs, stride_oh, stride_od,
        stride_out_logexpsum_b, stride_out_logexpsum_h,
        BLOCK_SEQ, BLOCK_DMODEL
    )
    
    return O, out_logexpsum
