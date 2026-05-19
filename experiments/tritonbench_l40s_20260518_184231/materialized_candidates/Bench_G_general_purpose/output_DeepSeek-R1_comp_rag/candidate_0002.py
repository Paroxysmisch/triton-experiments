import triton
import triton.language as tl
import torch

@triton.jit
def kldivergence_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute KL divergence: x * log(x / y)
    ratio = x / y
    log_ratio = tl.log(ratio)
    output = x * log_ratio
    
    tl.store(output_ptr + offsets, output, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape, "Input shapes must match"
    
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Adjust BLOCK_SIZE based on hardware constraints; 1024 is a common starting point
    kldivergence_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output
