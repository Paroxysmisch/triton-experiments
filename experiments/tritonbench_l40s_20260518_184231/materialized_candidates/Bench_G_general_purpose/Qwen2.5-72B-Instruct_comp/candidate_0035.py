import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start_ptr,  # Pointer to the array of starting indices
    old_a_len_ptr,    # Pointer to the array of lengths
    old_a_location,   # Pointer to the source array
    new_a_start,      # Pointer to the destination array
    new_a_location,   # Pointer to the destination array
    num_segments,     # Number of segments to copy
    BLOCK_SIZE: tl.constexpr
):
    # Get the segment ID for this program instance
    a_id = tl.program_id(0)
    
    # Check if the segment ID is within bounds
    if a_id < num_segments:
        # Load the starting index and length for the current segment
        old_start = tl.load(old_a_start_ptr + a_id)
        old_len = tl.load(old_a_len_ptr + a_id)
        
        # Calculate the new starting index in the destination array
        new_start = new_a_start + old_start
        
        # Iterate over the elements in the current segment
        for i in range(0, old_len, BLOCK_SIZE):
            # Compute the current block size (may be less than BLOCK_SIZE for the last block)
            block_size = min(BLOCK_SIZE, old_len - i)
            
            # Compute the offsets for the current block
            old_offset = old_start + i
            new_offset = new_start + i
            
            # Load the block from the source array
            old_block = tl.load(old_a_location + old_offset, mask=tl.arange(0, block_size) < block_size)
            
            # Store the block to the destination array
            tl.store(new_a_location + new_offset, old_block, mask=tl.arange(0, block_size) < block_size)

import torch

def launch_var_len_copy_triton(
    old_a_start,  # Tensor of starting indices
    old_a_len,    # Tensor of lengths
    old_a,        # Source tensor
    new_a,        # Destination tensor
    num_segments, # Number of segments to copy
    BLOCK_SIZE=256
):
    # Convert tensors to pointers
    old_a_start_ptr = old_a_start.to(device='cuda').contiguous().view(-1).cuda()
    old_a_len_ptr = old_a_len.to(device='cuda').contiguous().view(-1).cuda()
    old_a_location = old_a.to(device='cuda').contiguous().view(-1).cuda()
    new_a_location = new_a.to(device='cuda').contiguous().view(-1).cuda()
    
    # Launch the kernel
    grid = (num_segments, 1, 1)
    var_len_copy_kernel_triton[grid](old_a_start_ptr, old_a_len_ptr, old_a_location, 0, new_a_location, num_segments, BLOCK_SIZE)
