import triton
import triton.language as tl

@triton.jit
def erfc_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE=256):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Approximation of erfc using Taylor series
    c0 = 0.842700792949715
    c1 = -0.171204763999168
    c2 = 0.00783279472176
    c3 = -0.0002275014398
    c4 = 0.00000220745
    
    y = tl.where(x >= 0, x, -x)
    z = 1 / (1 + 0.5 * y)
    t = 1 - z * (c0 + z * (c1 + z * (c2 + z * (c3 + z * c4))))
    t = tl.where(x >= 0, t, 2 - t)
    
    tl.store(output_ptr + offsets, t, mask=mask)
