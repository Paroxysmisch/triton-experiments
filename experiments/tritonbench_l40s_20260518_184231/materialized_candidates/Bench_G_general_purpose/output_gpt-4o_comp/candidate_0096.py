import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _rms_norm_fwd_fused(X, W, Y, N, stride_x, stride_w, stride_y, BLOCK_SIZE: tl.constexpr):
    # Pointers for this program instance
    pid = tl.program_id(axis=0)
    
    # Offset to load the row
    row_start = pid * BLOCK_SIZE
    
    # Create a mask to handle the case when N is not a multiple of BLOCK_SIZE
    mask = row_start + tl.arange(0, BLOCK_SIZE) < N
    
    # Load a block of input data
    x_ptrs = X + row_start * stride_x + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute the square of each element
    x_squared = x * x
    
    # Sum the squares
    sum_x_squared = tl.sum(x_squared, axis=0)
    
    # Compute the mean of squares
    mean_x_squared = sum_x_squared / N
    
    # Compute the reciprocal of the standard deviation
    rstd = tl.rsqrt(mean_x_squared)
    
    # Normalize the input
    x_norm = x * rstd
    
    # Load the weights
    w_ptrs = W + row_start * stride_w + tl.arange(0, BLOCK_SIZE)
    weight = tl.load(w_ptrs, mask=mask, other=0.0)
    
    # Apply the weights
    y = x_norm * weight
    
    # Store the result
    y_ptrs = Y + row_start * stride_y + tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptrs, y, mask=mask)

def rmsnorm_forward(x, weight):
    # Get the dimensions of the input tensor
    M, N = x.shape
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Define the number of warps
    num_warps = 4  # This can be tuned for better performance based on your hardware
    
    # Define grid size
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    _rms_norm_fwd_fused[grid](
        x, weight, y, N,
        x.stride(0), weight.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return y

# Example usage
x = torch.randn(1024, 256, device='cuda')
weight = torch.randn(256, device='cuda')
y = rmsnorm_forward(x, weight)
