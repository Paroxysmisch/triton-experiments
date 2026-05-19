import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen, Mid_O, Mid_O_LogExpSum, O,
    stride_bs_b, stride_bs_s,
    stride_mo_b, stride_mo_h, stride_mo_s, stride_mo_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Extract batch and head indices
    batch_id = pid // stride_bs_b
    head_id = (pid % stride_bs_b) // stride_bs_s
    
    # Compute sequence length for this batch
    seq_len = tl.load(B_Seqlen + batch_id)
    
    # Create offsets for the current block
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_s = tl.arange(0, BLOCK_SEQ)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over sequence length in blocks
    for seq_idx in range(0, seq_len, BLOCK_SEQ):
        # Compute actual sequence length for this block
        block_seq = tl.minimum(BLOCK_SEQ, seq_len - seq_idx)
        
        # Load Mid_O block
        mid_o_ptrs = Mid_O + (
            batch_id * stride_mo_b +
            head_id * stride_mo_h +
            (seq_idx + offs_s[:block_seq]) * stride_mo_s +
            offs_d[None, :] * stride_mo_d
        )
        mid_o_block = tl.load(mid_o_ptrs)
        
        # Load LogExpSum for normalization
        log_exp_sum_ptr = Mid_O_LogExpSum + batch_id * stride_bs_b + head_id * stride_bs_s
        log_exp_sum = tl.load(log_exp_sum_ptr)
        
        # Normalize and accumulate
        mid_o_block = tl.exp(mid_o_block - log_exp_sum)
        acc += tl.sum(mid_o_block, axis=0)
    
    # Store result
    o_ptr = O + (
        batch_id * stride_o_b +
        head_id * stride_o_h +
        offs_d * stride_o_d
    )
    tl.store(o_ptr, acc)

def flash_decode_stage2(b_seqlen, mid_o, mid_o_logexpsum, BLOCK_SEQ=128, BLOCK_DMODEL=128):
    """
    Wrapper function for the flash decode stage2 kernel
    
    Args:
        b_seqlen: Tensor of sequence lengths for each batch [B]
        mid_o: Intermediate output tensor [B, H, S, D]
        mid_o_logexpsum: Log sum of exponentials [B, H]
        BLOCK_SEQ: Sequence dimension block size
        BLOCK_DMODEL: Model dimension block size
    
    Returns:
        O: Output tensor [B, H, D]
    """
    batch_size, num_heads, max_seq_len, d_model = mid_o.shape
    
    # Create output tensor
    o = torch.empty((batch_size, num_heads, d_model), 
                   device=mid_o.device, dtype=mid_o.dtype)
    
    # Calculate strides
    stride_bs_b = num_heads
    stride_bs_s = 1
    
    stride_mo_b = num_heads * max_seq_len * d_model
    stride_mo_h = max_seq_len * d_model
    stride_mo_s = d_model
    stride_mo_d = 1
    
    stride_o_b = num_heads * d_model
    stride_o_h = d_model
    stride_o_s = 1
    stride_o_d = 1
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel_flash_decode_stage2[grid](
        b_seqlen, mid_o, mid_o_logexpsum, o,
        stride_bs_b, stride_bs_s,
        stride_mo_b, stride_mo_h, stride_mo_s, stride_mo_d,
        stride_o_b, stride_o_h, stride_o_s, stride_o_d,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return o
