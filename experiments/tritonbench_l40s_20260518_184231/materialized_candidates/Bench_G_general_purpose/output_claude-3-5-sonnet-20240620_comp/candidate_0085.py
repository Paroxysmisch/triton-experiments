import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    # Pointers to tensors
    Mid_O_ptr, Mid_O_LogExpSum_ptr, O_ptr, B_Seqlen_ptr,
    # Dimensions
    batch, head, seq_block_num, head_dim,
    # Strides for Mid_O
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    # Strides for Mid_O_LogExpSum
    stride_les_ob, stride_les_oh, stride_les_os,
    # Strides for output O
    stride_o_ob, stride_o_oh, stride_o_od,
    # Constants
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Get program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Compute the sequence length for this batch
    seqlen = tl.load(B_Seqlen_ptr + pid_batch)
    
    # Initialize accumulator for this batch and head
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    max_logit = float("-inf")
    
    # Create offsets for vectorized loads
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Base pointers for this batch and head
    mid_o_base = Mid_O_ptr + pid_batch * stride_mid_ob + pid_head * stride_mid_oh
    les_base = Mid_O_LogExpSum_ptr + pid_batch * stride_les_ob + pid_head * stride_les_oh
    
    # Loop over sequence blocks
    for seq_idx in range(0, seq_block_num):
        # Load log-exp-sum for this block
        block_les = tl.load(les_base + seq_idx * stride_les_os)
        
        # Update max logit for numerical stability
        max_logit = tl.maximum(max_logit, block_les)
        
        # Load Mid_O values for this block
        mid_o_ptrs = mid_o_base + seq_idx * stride_mid_os + offs_d * stride_mid_od
        block_values = tl.load(mid_o_ptrs, mask=offs_d < head_dim)
        
        # Compute and accumulate weighted values
        exp_scale = tl.exp(block_les - max_logit)
        acc += block_values * exp_scale
    
    # Normalize the accumulated values
    final_scale = tl.exp(-max_logit)
    acc *= final_scale
    
    # Write output
    o_base = O_ptr + pid_batch * stride_o_ob + pid_head * stride_o_oh
    o_ptrs = o_base + offs_d * stride_o_od
    tl.store(o_ptrs, acc, mask=offs_d < head_dim)

def flash_decode_stage2(Mid_O, Mid_O_LogExpSum, B_Seqlen, O):
    """
    Wrapper function for the flash decode stage2 kernel.
    
    Args:
        Mid_O: tensor of shape [batch, head, seq_block_num, head_dim]
        Mid_O_LogExpSum: tensor of shape [batch, head, seq_block_num]
        B_Seqlen: tensor of shape [batch]
        O: output tensor of shape [batch, head, head_dim]
    """
    batch, head, seq_block_num, head_dim = Mid_O.shape
    
    # Constants for block sizes
    BLOCK_SEQ = 32
    BLOCK_DMODEL = 32
    
    # Ensure head_dim is compatible with block size
    assert head_dim % BLOCK_DMODEL == 0, f"head_dim {head_dim} must be divisible by BLOCK_DMODEL {BLOCK_DMODEL}"
    
    # Launch kernel
    grid = (batch, head)
    _fwd_kernel_flash_decode_stage2[grid](
        Mid_O, Mid_O_LogExpSum, O, B_Seqlen,
        batch, head, seq_block_num, head_dim,
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1), Mid_O_LogExpSum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return O
