import triton
import triton.language as tl
import torch

# Triton kernel for scaling and adding vectors
@triton.jit
def scaled_add_kernel(y_ptr, x_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from global memory
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the operation y += alpha * x
    y += alpha * x
    
    # Store the result back to global memory
    tl.store(y_ptr + offsets, y, mask=mask)

# Wrapper function
def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "Input tensors must have the same shape"
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    scaled_add_kernel[grid](y, x, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute the dot product of the modified y with itself
    dot_product = torch.dot(y, y)
    
    return dot_product

# Example usage
y = torch.randn(1024, device='cuda')
x = torch.randn(1024, device='cuda')
alpha = 0.5
result = scaled_add_dot(y, x, alpha)
print(result)
