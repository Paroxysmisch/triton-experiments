import triton
import triton.language as tl
import torch

@triton.jit
def pairwise_distance_kernel(x1, x2, p_distance, eps_distance, output, n):
    # Compute pairwise distance
    row = tl.arange(0, n)
    col = tl.arange(0, n)
    x1_row = x1[row, :]
    x2_col = x2[col, :]
    
    # Calculate the distance using the specified norm
    distance = tl.norm(x1_row - x2_col, p_distance, dim=1) + eps_distance
    output[row, col] = distance

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    # Ensure x1 and x2 have the same shape
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    
    # Get the number of elements
    n = x1.shape[0]
    
    # Allocate output tensor
    output = torch.empty((n, n), device=x1.device, dtype=x1.dtype)
    
    # Launch the kernel
    pairwise_distance_kernel[(n,)](x1, x2, p_distance, eps_distance, output, n)
    
    # Normalize the distances
    max_distance = torch.max(output, dim=dim_norm, keepdim=keepdim)[0] + eps_norm
    normalized_distance = output / max_distance
    
    return normalized_distance
