import triton
import triton.language as tl
import torch

# Triton kernel function
@triton.jit
def scaled_add_dot_kernel(y_ptr, x_ptr, alpha, BLOCK_SIZE: int):
    """
    This kernel performs the operation y += alpha * x and computes the dot product of the modified y with itself.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < y_ptr.size

    # Load data from global memory
    y = tl.load(y_ptr + offset, mask=mask)
    x = tl.load(x_ptr + offset, mask=mask)

    # Perform y += alpha * x
    y += alpha * x

    # Store the result back to global memory
    tl.store(y_ptr + offset, y, mask=mask)

    # Compute the dot product of y with itself
    y_squared = y * y
    dot_product = tl.sum(y_squared, axis=0)

    # Write the dot product to the output location
    tl.atomic_add(y_ptr + 0, dot_product, mask=mask)

# Wrapper function to launch the Triton kernel
def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes y += alpha * x and returns the dot product of the modified y with itself.
    
    Parameters:
    y (Tensor): The target tensor to be modified, of shape (n,).
    x (Tensor): The tensor to be scaled and added to y, of shape (n,).
    alpha (float): The scalar multiplier for x.
    
    Returns:
    Tensor: The dot product of the modified y with itself.
    """
    # Ensure the tensors are on the same device
    assert y.device == x.device, "Tensors must be on the same device"
    
    # Ensure the tensors have the same shape
    assert y.shape == x.shape, "Tensors must have the same shape"
    
    # Allocate a tensor to store the dot product
    dot_product = torch.zeros(1, device=y.device, dtype=y.dtype)
    
    # Define the block size and grid dimensions
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    scaled_add_dot_kernel[grid](y, x, alpha, BLOCK_SIZE)
    
    # Return the dot product
    return dot_product

# Example usage
y = torch.tensor([1.0, 2.0, 3.0], device='cuda')
x = torch.tensor([4.0, 5.0, 6.0], device='cuda')
alpha = 2.0
result = scaled_add_dot(y, x, alpha)
print("Modified y:", y)
print("Dot product:", result)
