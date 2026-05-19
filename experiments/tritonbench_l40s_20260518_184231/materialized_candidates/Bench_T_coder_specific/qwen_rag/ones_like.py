import triton
import triton.language as tl

@triton.jit
def ones_kernel(
    output_ptr,  # Pointer to the output tensor in GPU memory
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Size of each block of threads
):
    pid = tl.program_id(axis=0)  # Get the block index
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets for each thread
    mask = offsets < n_elements  # Mask to ensure we don't write out of bounds
    tl.store(output_ptr + offsets, 1.0, mask=mask)  # Store 1.0 in all valid positions
