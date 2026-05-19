import torch
import triton
import triton.language as tl

# Triton kernel for log1p
@triton.jit
def log1p_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to filter out out-of-bounds indices
    mask = offsets < n_elements
    # Load the input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask)
    # Compute log1p
    output_elements = tl.log1p(input_elements)
    # Store the results
    tl.store(output_ptr + offsets, output_elements, mask=mask)

# Wrapper function for log1p
def log1p(input, *, out=None):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    # Enqueue the kernel
    log1p_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        out.data_ptr(),
        input.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
