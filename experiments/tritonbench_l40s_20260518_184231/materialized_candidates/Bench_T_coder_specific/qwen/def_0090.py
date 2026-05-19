import triton
import triton.language as tl

@triton.jit
def fused_dropout_hardshrink_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    lambd,
    training,
    inplace,
    block_size: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    if training:
        rand_val = tl.random.rand() < p
        x = tl.where(rand_val, tl.zeros_like(x), x)
    
    x = tl.where(tl.abs(x) <= lambd, tl.zeros_like(x), x)
    
    if inplace:
        tl.store(output_ptr + offsets, x, mask=mask)
    else:
        tl.store(output_ptr + offsets, x, mask=mask)
