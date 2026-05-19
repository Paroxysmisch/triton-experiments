import torch
import triton
import triton.language as tl

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
    # Note: Ensure x and y contain positive values to avoid numerical issues
    ratio = tl.where(x > 0, x / y, 0.0)
    log_ratio = tl.where(x > 0, tl.log(ratio), 0.0)
    output = x * log_ratio
    
    tl.store(output_ptr + offsets, output, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor):
    """
    Compute element-wise Kullback-Leibler divergence between two tensors.
    
    Args:
        x: Input tensor (must be same shape as y and on GPU)
        y: Input tensor (must be same shape as x and on GPU)
    
    Returns:
        torch.Tensor: Tensor containing element-wise KL divergence values
    """
    # Input validation
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Grid function calculates number of blocks needed
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with optimal block size for typical cases
    kldivergence_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=1024,
        num_warps=4
    )
    
    return output

x = torch.rand(1000000, device='cuda').abs() + 1e-6
y = torch.rand(1000000, device='cuda').abs() + 1e-6
output = kldivergence(x, y)
