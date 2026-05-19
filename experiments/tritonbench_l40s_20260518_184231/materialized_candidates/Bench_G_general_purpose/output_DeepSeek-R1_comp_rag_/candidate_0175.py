import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    Logits,  # Pointer to the logits tensor
    Out,     # Pointer to the output tensor
    B_Start_Loc,  # Pointer to the start locations for each batch
    B_Seqlen,      # Pointer to the sequence lengths for each batch
    max_input_len,  # Maximum sequence length in the batch
    stride_logits_h,  # Stride for the head dimension in logits
    stride_logits_s,  # Stride for the sequence (token) dimension in logits
    stride_out_h,     # Stride for the head dimension in output
    stride_out_s,    # Stride for the sequence dimension in output
    BLOCK_SIZE: tl.constexpr,  # Block size for processing elements
):
    # Extract program IDs for batch and head
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Load the start location and sequence length for the current batch
    cur_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    
    # Compute pointers for the current head in logits and output
    logits_ptr = Logits + cur_head * stride_logits_h
    out_ptr = Out + cur_head * stride_out_h
    
    # Initialize max_logit to negative infinity and sum_exp to zero
    max_logit = float("-inf")
    sum_exp = 0.0
    
    # First pass: Compute the maximum logit value
    for block_start in range(0, cur_seq_len, BLOCK_SIZE):
        # Calculate offsets and mask for valid elements
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < cur_seq_len
        
        # Load logits for the current block, applying mask
        logits = tl.load(
            logits_ptr + (cur_start_loc + offs) * stride_logits_s,
            mask=mask,
            other=float("-inf"),
        )
        # Update max_logit with the maximum value found in this block
        curr_max = tl.max(logits, axis=0)
        max_logit = tl.maximum(max_logit, curr_max)
    
    # Second pass: Compute the sum of exponentials
    for block_start in range(0, cur_seq_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < cur_seq_len
        
        logits = tl.load(
            logits_ptr + (cur_start_loc + offs) * stride_logits_s,
            mask=mask,
            other=0.0,
        )
        # Compute exponentials adjusted by max_logit and accumulate sum
        exp_logits = tl.exp(logits - max_logit)
        sum_exp += tl.sum(exp_logits, axis=0)
    
    # Third pass: Compute softmax values and store the result
    for block_start in range(0, cur_seq_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < cur_seq_len
        
        logits = tl.load(
            logits_ptr + (cur_start_loc + offs) * stride_logits_s,
            mask=mask,
            other=0.0,
        )
        # Calculate softmax values and store them
        exp_logits = tl.exp(logits - max_logit)
        softmax_val = exp_logits / sum_exp
        tl.store(
            out_ptr + (cur_start_loc + offs) * stride_out_s,
            softmax_val,
            mask=mask,
        )

@torch.no_grad()
def token_softmax_fwd(logits, out, b_start_loc, b_seq_len, max_input_len):
    # Ensure that the sequence length does not exceed max_input_len
    assert b_seq_len.max() <= max_input_len, "Sequence length exceeds max_input_len"
    
    batch_size = b_seq_len.shape[0]
    num_heads = logits.size(0)  # Logits shape: [num_heads, total_tokens]
    
    # Set block size as the next power of two less than max_input_len, capped at 4096
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    # Adjust the number of warps based on the block size for optimization
    num_warps = 4
    if BLOCK_SIZE <= 2048:
        num_warps = 2
    if BLOCK_SIZE <= 512:
        num_warps = 1
    
    # Define the grid with dimensions (batch_size, num_heads)
    grid = (batch_size, num_heads)
    
    # Launch the kernel with configured parameters
    _fwd_kernel_token_softmax[grid](
        logits,
        out,
        b_start_loc,
        b_seq_len,
        max_input_len,
        logits.stride(0),  # stride_logits_h
        logits.stride(1),  # stride_logits_s
        out.stride(0),     # stride_out_h
        out.stride(1),     # stride_out_s
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
