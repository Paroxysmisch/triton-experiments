import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen, # [batch]
    Mid_O,    # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,        # [batch, head, head_dim]
    out_logexpsum,  # [batch, head]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_out_logexpsum_b, stride_out_logexpsum_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Get the current batch and head indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Create offset for the head dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load sequence length for current batch
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    
    # Calculate number of blocks needed for current sequence
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, 
                           (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)
    
    # Initialize accumulators
    sum_exp = 0.0
    max_logic = float("-1e20")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Calculate base offsets for Mid_O and Mid_O_LogExpSum
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    
    # Main computation loop
    for block_seq_n in range(0, block_n_size):
        # Load values and logits
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n * stride_mid_o_es)
        
        # Update maximum logic value
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        # Scale previous accumulator
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        
        # Compute and accumulate weighted values
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    # Store results only if we processed any blocks
    if block_n_size > 0:
        # Store normalized output
        tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, 
                acc / sum_exp)
        # Store log-sum-exp
        tl.store(out_logexpsum + cur_batch * stride_out_logexpsum_b + 
                cur_head * stride_out_logexpsum_h,
                max_logic + tl.log(sum_exp))

def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, out_logexpsum, 
                       BLOCK_SEQ=128, BLOCK_DMODEL=128):
    """
    Wrapper function to launch the flash decode stage 2 kernel
    
    Parameters:
        B_Seqlen: torch.Tensor - Batch sequence lengths
        Mid_O: torch.Tensor - Intermediate output tensor [batch, head, seq_block_num, head_dim]
        Mid_O_LogExpSum: torch.Tensor - Log-sum-exp values [batch, head, seq_block_num]
        O: torch.Tensor - Output tensor [batch, head, head_dim]
        out_logexpsum: torch.Tensor - Output log-sum-exp [batch, head]
        BLOCK_SEQ: int - Sequence block size
        BLOCK_DMODEL: int - Model dimension block size
    """
    batch_size = Mid_O.shape[0]
    num_heads = Mid_O.shape[1]
    
    # Calculate strides
    stride_mid_ob = Mid_O.stride(0)
    stride_mid_oh = Mid_O.stride(1)
    stride_mid_os = Mid_O.stride(2)
    stride_mid_od = Mid_O.stride(3)
    
    stride_mid_o_eb = Mid_O_LogExpSum.stride(0)
    stride_mid_o_eh = Mid_O_LogExpSum.stride(1)
    stride_mid_o_es = Mid_O_LogExpSum.stride(2)
    
    stride_obs = O.stride(0)
    stride_oh = O.stride(1)
    stride_od = O.stride(2)
    
    stride_out_logexpsum_b = out_logexpsum.stride(0)
    stride_out_logexpsum_h = out_logexpsum.stride(1)
    
    # Launch kernel
    grid = (batch_size, num_heads)
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, O, out_logexpsum,
        stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
        stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
        stride_obs, stride_oh, stride_od,
        stride_out_logexpsum_b, stride_out_logexpsum_h,
        BLOCK_SEQ, BLOCK_DMODEL
    )
