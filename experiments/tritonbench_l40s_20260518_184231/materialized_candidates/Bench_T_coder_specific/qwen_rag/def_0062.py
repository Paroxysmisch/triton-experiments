import triton
import torch

# Define the Triton kernel function to perform the scaled add operation
@triton.jit
def scaled_add_dot_kernel(y_ptr, x_ptr, n, alpha, BLOCK_SIZE: int):
    """
    This kernel function scales the tensor x by a factor of alpha and adds it to y.
    Then, it computes the dot product of the modified y with itself.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load elements from y and x within bounds
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Scale x by alpha and add to y
    y_scaled = y + alpha * x
    
    # Store the updated values back to y
    tl.store(y_ptr + offsets, y_scaled, mask=mask)
    
    # Compute the dot product of the modified y with itself
    y_scaled = tl.where(mask, y_scaled, 0.0)
    local_sum = tl.sum(tl.square(y_scaled))
    total_sum = tl.program_barrier()
    if pid == 0:
        return total_sum

# Wrapper function to launch the Triton kernel and compute the dot product
def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    n = y.shape[0]
    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    result = scaled_add_dot_kernel[grid](y, x, n, alpha, BLOCK_SIZE)
    
    # Return the computed dot product
    return result

# Example usage
if __name__ == "__main__":
    y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    x = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    alpha = 2.0
    result = scaled_add_dot(y, x, alpha)
    print("Dot Product:", result.item())
