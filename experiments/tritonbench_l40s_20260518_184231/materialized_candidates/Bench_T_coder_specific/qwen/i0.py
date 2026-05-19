import triton
import triton.language as tl

@triton.jit
def bessel_i0_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    x = tl.load(input_ptr + indices, mask=mask)
    
    # Initialize variables
    sum_val = tl.zeros_like(x)
    term = tl.ones_like(x)
    k_fact_squared = tl.ones_like(x)
    
    # Compute the series expansion up to a reasonable number of terms
    max_terms = 100
    for k in range(max_terms):
        term *= (x * x / 4) / ((k + 1) * (k + 1))
        k_fact_squared *= (k + 1) * (k + 1)
        sum_val += term / k_fact_squared
    
    tl.store(output_ptr + indices, sum_val, mask=mask)
