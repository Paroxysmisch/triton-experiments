import torch
import triton
import triton.language as tl

# Triton kernel to compute gammaln
@triton.jit
def gammaln_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    # Compute gammaln
    output_vec = tl.log(tl.abs(tl.gamma(input_vec)))
    tl.store(output_ptr + offsets, output_vec, mask=mask)

# Wrapper function for gammaln
def gammaln(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024  # Adjust block size as needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    gammaln_kernel[grid, BLOCK_SIZE](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
