import triton
import triton.language as tl

@triton.jit
def sub_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)

    # Perform the subtraction with scaling
    result = input_val - alpha * other_val

    # Store the result back to memory
    tl.store(output_ptr + offsets, result, mask=mask)
