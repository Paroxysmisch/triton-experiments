import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_kernel(X_ptr, W_ptr, Y_ptr, N, stride_xm, stride_wn, BLOCK_SIZE: tl.constexpr):
    # Create a program index for each block
    pid = tl.program_id(axis=0)
    
    # Calculate the starting position for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds access
    mask = offs < N
    
    # Load input values
    x = tl.load(X_ptr + offs * stride_xm, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    var = tl.sum((x - mean) * (x - mean), axis=0) / N
    
    # Normalize
    norm_x = (x - mean) / tl.sqrt(var + 1e-5)
    
    # Load weights
    w = tl.load(W_ptr + offs * stride_wn, mask=mask, other=1.0)
    
    # Apply weights
    y = norm_x * w
    
    # Store result
    tl.store(Y_ptr + offs * stride_xm, y, mask=mask)

def layernorm_forward(X, W):
    # Get dimensions
    M, N = X.shape
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Get strides
    stride_xm = X.stride(0)
    stride_wn = W.stride(0)
    
    # Define block size
    BLOCK_SIZE = 1024  # This can be tuned for performance
    
    # Calculate number of blocks
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    _layer_norm_fwd_kernel[grid](X, W, Y, N, stride_xm, stride_wn, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y

# Example usage
M, N = 64, 1024  # Example dimensions
X = torch.randn(M, N, device='cuda')  # Input tensor
W = torch.ones(N, device='cuda')  # Weight tensor
Y = layernorm_forward(X, W)
