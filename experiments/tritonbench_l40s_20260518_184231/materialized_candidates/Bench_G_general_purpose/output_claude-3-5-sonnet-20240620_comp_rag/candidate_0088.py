import triton
import triton.language as tl
import torch

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
    BLOCK_DMODEL: tl.constexpr):
    
    # Get current batch and head indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Create offset for dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load sequence length for current batch
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    
    # Calculate number of blocks needed
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, 
                           (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)
    
    # Initialize accumulators
    sum_exp = 0.0
    max_logic = float("-1e20")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Calculate base offsets
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    
    # Main computation loop
    for block_seq_n in range(0, block_n_size):
        # Load values and logits
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n * stride_mid_o_es)
        
        # Update max logic and scale
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        
        # Scale accumulator and update
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        
        # Update sum of exponentials
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    # Store results if we have valid blocks
    if block_n_size > 0:
        out_ptr = O + cur_batch * stride_obs + cur_head * stride_oh + offs_d
        tl.store(out_ptr, acc / sum_exp)
        
        logexpsum_ptr = out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h
        tl.store(logexpsum_ptr, max_logic + tl.log(sum_exp))

def flash_decode_stage2(b_seqlen, mid_o, mid_o_logexpsum, BLOCK_SEQ=128, BLOCK_DMODEL=128):
    """
    Wrapper function for the flash decode stage 2 kernel.
    
    Args:
        b_seqlen: Tensor of sequence lengths for each batch [batch]
        mid_o: Input tensor [batch, head, seq_block_num, head_dim]
        mid_o_logexpsum: Input tensor [batch, head, seq_block_num]
        BLOCK_SEQ: Sequence block size
        BLOCK_DMODEL: Model dimension block size
    
    Returns:
        o: Output tensor [batch, head, head_dim]
        out_logexpsum: Output tensor [batch, head]
    """
    batch, head, seq_block_num, head_dim = mid_o.shape
    
    # Create output tensors
    o = torch.empty((batch, head, head_dim), 
                   device=mid_o.device, dtype=mid_o.dtype)
    out_logexpsum = torch.empty((batch, head), 
                               device=mid_o.device, dtype=mid_o.dtype)
    
    # Calculate strides
    stride_mid_ob = mid_o.stride(0)
    stride_mid_oh = mid_o.stride(1)
    stride_mid_os = mid_o.stride(2)
    stride_mid_od = mid_o.stride(3)
    
    stride_mid_o_eb = mid_o_logexpsum.stride(0)
    stride_mid_o_eh = mid_o_logexpsum.stride(1)
    stride_mid_o_es = mid_o_logexpsum.stride(2)
    
    stride_obs = o.stride(0)
    stride_oh = o.stride(1)
    stride_od = o.stride(2)
    
    stride_out_logexpsum_b = out_logexpsum.stride(0)
    stride_out_logexpsum_h = out_logexpsum.stride(1)
    
    # Launch kernel
    grid = (batch, head)
    _fwd_kernel_flash_decode_stage2[grid](
        b_seqlen, mid_o, mid_o_logexpsum, o, out_logexpsum,
        stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
        stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
        stride_obs, stride_oh, stride_od,
        stride_out_logexpsum_b, stride_out_logexpsum_h,
        BLOCK_SEQ, BLOCK_DMODEL
    )
    
    return o, out_logexpsum
