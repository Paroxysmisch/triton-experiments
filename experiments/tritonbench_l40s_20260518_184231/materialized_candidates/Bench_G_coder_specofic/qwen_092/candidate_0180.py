import torch
import triton
import triton.language as tl

# Triton kernel definition
_l2_norm_fwd_1pass_kernel = triton.compile(_l2_norm_fwd_1pass_kernel, 
                                         signature=triton.signature([triton.f32, triton.f32, triton.i32, triton.i32, triton.f32, triton.constexpr],
                                                                     [triton.f32], 
                                                                     num_warps=4))

def _l2_norm_fwd(x, eps=1e-8):
    """
    Perform L2 normalization on a 2D input tensor `x`.
    """
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    
    # Get the number of rows (M) and columns (N)
    M, N = x.shape
    
    # Calculate BLOCK_N based on the element size and ensure it doesn't exceed 64KB
    element_size = x.element_size()
    BLOCK_N = min(64 * 1024 // element_size, N)
    
    # Check if the feature dimension is larger than BLOCK_N
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension N ({N}) is larger than BLOCK_N ({BLOCK_N})")
    
    # Initialize an empty tensor `y` to store the output
    y = torch.empty_like(x)
    
    # Launch the kernel
    _l2_norm_fwd_1pass_kernel[(M,)](x, y, stride_x_row=x.stride(0), N=N, eps=eps, BLOCK_N=BLOCK_N)
    
    return y
