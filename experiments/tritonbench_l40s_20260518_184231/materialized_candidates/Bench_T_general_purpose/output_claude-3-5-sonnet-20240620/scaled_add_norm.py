import triton
import triton.language as tl

@triton.jit
def scaled_add_norm_kernel(y_ptr, x_ptr, alpha, n_elements):
    # Get the index of the current thread
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    
    # Load the elements of y and x
    y = tl.load(y_ptr + idx)
    x = tl.load(x_ptr + idx)
    
    # Perform the scaled addition
    y += alpha * x
    
    # Store the updated value back to y
    tl.store(y_ptr + idx, y)
    
    # Compute the 2-norm
    norm = tl.sqrt(tl.sum(y * y))
    
    return norm

import torch

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> float:
    # Ensure the input tensors are of the correct shape
    assert y.shape == x.shape, "Shapes of y and x must match."
    assert y.ndim == 1, "y must be a 1D tensor."
    assert x.ndim == 1, "x must be a 1D tensor."
    
    # Get the number of elements
    n_elements = y.numel()
    
    # Launch the Triton kernel
    norm = scaled_add_norm_kernel[(n_elements + 255) // 256](y.data_ptr(), x.data_ptr(), alpha, n_elements)
    
    return norm.item()
