import triton
import torch
import triton.language as tl

# Triton kernel to perform the scaled addition
@triton.jit
def scaled_add_kernel(y_ptr, x_ptr, alpha, n_elements, BLOCK_SIZE: int):
    """
    Kernel that computes y += alpha * x for two vectors y and x.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load data from global memory
    y = tl.load(y_ptr + offset)
    x = tl.load(x_ptr + offset)
    
    # Perform the scaled addition
    y += alpha * x
    
    # Store the result back to global memory
    tl.store(y_ptr + offset, y)

# Wrapper function to launch the Triton kernel and compute the 2-norm
def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> float:
    """
    Modifies y in place by adding alpha * x and returns the 2-norm of the modified y.
    """
    assert y.shape == x.shape, "Shapes of y and x must match."
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scaled_add_kernel[grid](y, x, alpha, n_elements, BLOCK_SIZE)
    
    # Calculate and return the 2-norm of the modified y
    return torch.norm(y, p=2)
