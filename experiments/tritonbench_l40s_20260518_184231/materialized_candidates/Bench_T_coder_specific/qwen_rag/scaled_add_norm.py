import triton
import triton.language as tl

@triton.jit
def scaled_add_norm_kernel(y_ptr, x_ptr, alpha, n_elements: tl.int32):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    y += alpha * x
    
    tl.store(y_ptr + offsets, y, mask=mask)
