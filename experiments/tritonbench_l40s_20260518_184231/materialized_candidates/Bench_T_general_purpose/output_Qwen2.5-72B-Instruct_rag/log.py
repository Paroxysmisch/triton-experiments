import torch
import triton
import triton.language as tl

# Triton kernel for computing the natural logarithm
@triton.jit
def log_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the 1D grid of program IDs
    pid = tl.program_id(0)
    # Compute the start index for this program instance
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't access out-of-bounds memory
    mask = offsets < n_elements
    # Load the input elements
    input_elements = tl.load(input_ptr + offsets, mask=mask)
    # Compute the natural logarithm
    output_elements = tl.log(input_elements)
    # Store the result back to the output tensor
    tl.store(output_ptr + offsets, output_elements, mask=mask)

# Wrapper function for the log operation
def log(input, *, out=None):
    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size (next power of two greater than the number of elements)
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    # Enqueue the kernel
    log_kernel[(n_elements // BLOCK_SIZE + (n_elements % BLOCK_SIZE > 0))](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
