import triton
import triton.language as tl

@triton.jit
def log_kernel(
    x_ptr: ptr[float32],
    y_ptr: ptr[float32],
    n_elements: int32,
    BLOCK_SIZE: int32 = 1024
):
    pid = triton.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = pid * BLOCK_SIZE + arange(BLOCK_SIZE)
    mask = offsets < n_elements
    x = x_ptr[offsets][mask]
    y = y_ptr[offsets][mask]
    y[mask] = log(x)

def log(input, *, out=None):
    if out is None:
        out = input.clone()
    
    n_elements = input.numel()
    grid_size = ceil_div(n_elements, 1024)
    
    log_kernel[grid_size, 1024](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
