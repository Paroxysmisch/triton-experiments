import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n_elements, block_size)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Truncate the values
    y = tl.floor(x)
    
    tl.store(y_ptr + offsets, y, mask=mask)
