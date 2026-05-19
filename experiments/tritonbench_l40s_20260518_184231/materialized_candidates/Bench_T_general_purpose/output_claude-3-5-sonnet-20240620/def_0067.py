import triton
import triton.language as tl

@triton.jit
def fused_pairwise_distance_kernel(x1_ptr, x2_ptr, output_ptr, output_size, p, eps, n, m):
    # Calculate the adaptive average pooling
    # ... (implementation of adaptive_avg_pool2d in Triton)
    
    # Calculate pairwise distance
    for i in range(n):
        for j in range(m):
            # Compute the distance using the specified norm p
            # ... (distance calculation logic)
            output_ptr[i, j] = distance_value + eps  # Avoid division by zero

import torch

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Ensure input tensors are on the same device
    if x1.device != x2.device:
        raise ValueError("Input tensors must be on the same device.")
    
    # Get the shape of the input tensors
    n, c, h, w = x1.shape
    
    # Allocate output tensor
    output_shape = (n, n) if not keepdim else (n, n, 1)
    output = torch.empty(output_shape, device=x1.device)
    
    # Launch the Triton kernel
    fused_pairwise_distance_kernel[(n, n)](x1, x2, output, output_size, p, eps, n, c)
    
    return output
