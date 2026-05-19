import triton
import triton.language as tl

@triton.jit
def _log_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.log(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def log(input, *, out=None):
    if out is None:
        # Assuming the presence of a tensor-creation function "empty_like" 
        # which creates an uninitialized tensor with the same shape/type as input.
        # Replace "empty_like" with your actual method of allocating an output tensor.
        out = empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _log_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
