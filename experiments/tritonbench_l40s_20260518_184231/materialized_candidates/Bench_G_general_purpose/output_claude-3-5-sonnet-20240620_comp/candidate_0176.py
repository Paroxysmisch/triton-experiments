import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    output_ptr, # pointer to output tensor [B, H, S, S]
    input_ptr,  # pointer to input tensor [B, H, S, S]
    seqlen_ptr, # pointer to sequence lengths [B]
    batch_size, # batch size
    num_heads,  # number of attention heads
    max_seqlen, # maximum sequence length
    BLOCK_SIZE: tl.constexpr  # block size for parallelization
):
    # Program ID
    pid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # head index
    
    # Get sequence length for current batch
    seqlen = tl.load(seqlen_ptr + pid)
    
    # Compute row offset for current batch and head
    row_start = pid * num_heads * max_seqlen * max_seqlen + hid * max_seqlen * max_seqlen
    
    # Iterate over sequence length in blocks
    for start_idx in range(0, max_seqlen, BLOCK_SIZE):
        # Block size for current iteration
        block_size = min(BLOCK_SIZE, max_seqlen - start_idx)
        
        # Create offset array for current block
        offs = start_idx + tl.arange(0, block_size)
        
        # Skip if beyond actual sequence length
        if start_idx >= seqlen:
            continue
            
        # Load input values for current block
        row_offs = row_start + offs * max_seqlen
        block_mask = offs < seqlen
        
        # Initialize variables for max and sum computation
        max_val = tl.full([block_size], float("-inf"), dtype=tl.float32)
        
        # First pass: find max value
        for i in range(0, seqlen, BLOCK_SIZE):
            col_offs = tl.arange(0, BLOCK_SIZE)
            col_mask = col_offs < seqlen
            ptrs = row_offs[:, None] + col_offs[None, :]
            values = tl.load(input_ptr + ptrs, mask=col_mask[None, :], other=float("-inf"))
            max_val = tl.maximum(max_val, tl.max(values, 1))
            
        # Second pass: compute exponentials and sum
        exp_sum = tl.zeros([block_size], dtype=tl.float32)
        
        for i in range(0, seqlen, BLOCK_SIZE):
            col_offs = tl.arange(0, BLOCK_SIZE)
            col_mask = col_offs < seqlen
            ptrs = row_offs[:, None] + col_offs[None, :]
            values = tl.load(input_ptr + ptrs, mask=col_mask[None, :], other=float("-inf"))
            
            # Subtract max for numerical stability
            values = values - max_val[:, None]
            exp_values = tl.exp(values)
            exp_sum += tl.sum(exp_values, 1)
            
        # Final pass: compute softmax and write output
        for i in range(0, seqlen, BLOCK_SIZE):
            col_offs = tl.arange(0, BLOCK_SIZE)
            col_mask = col_offs < seqlen
            ptrs = row_offs[:, None] + col_offs[None, :]
            values = tl.load(input_ptr + ptrs, mask=col_mask[None, :], other=float("-inf"))
            
            # Compute softmax
            values = values - max_val[:, None]
            exp_values = tl.exp(values)
            softmax_values = exp_values / exp_sum[:, None]
            
            # Store results
            tl.store(output_ptr + ptrs, softmax_values, mask=col_mask[None, :])

# Wrapper function
def token_softmax_fwd(x, seqlens):
    batch_size, num_heads, max_seqlen, _ = x.shape
    output = torch.empty_like(x)
    
    # Determine optimal block size and number of warps
    BLOCK_SIZE = min(max_seqlen, 256)
    num_warps = 4 if BLOCK_SIZE > 128 else 2
    
    # Launch kernel
    grid = (batch_size, num_heads)
    _fwd_kernel_token_softmax[grid](
        output,
        x,
        seqlens,
        batch_size,
        num_heads,
        max_seqlen,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output
