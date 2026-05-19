import triton
import triton.language as tl

@triton.jit
def logspace_kernel(
    output_ptr,  # Pointer to the output tensor
    start,       # Starting value for the set of points
    end,         # Ending value for the set of points
    step_size,   # Step size between points
    base,        # Base of the logarithm function
    n_elements,  # Total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    pid = tl.program_id(axis=0)  # Program ID for the current block
    block_start = pid * BLOCK_SIZE  # Start index for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for the current block
    mask = offsets < n_elements  # Mask to ensure we don't write out of bounds
    
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
        exponent = start + i * step_size
        value = tl.math.pow(base, exponent)
        tl.store(output_ptr + idx, value, mask=mask)
