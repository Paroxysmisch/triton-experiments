import triton
import triton.language as tl
import torch

@triton.jit
def exp_sqrt_kernel(
    X: ptr[T],  # Input tensor
    Y: ptr[T],  # Output tensor
    N: int32,   # Number of elements in the input tensor
    BLOCK_SIZE: int32 = 1024
):
    pid = triton.program_id(0)
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Get the index of the current element
    i = pid * BLOCK_SIZE + triton.block_idx(0)
    
    if i < N:
        # Calculate the exponential of the input element
        x_exp = math.exp(X[i])
        
        # Check for overflow or underflow
        if x_exp == float('inf') or x_exp == float('-inf'):
            Y[i] = float('nan')
        else:
            # Calculate the square root of the exponential result
            y_val = math.sqrt(x_exp)
            Y[i] = y_val

def exp_sqrt(input, out=None) -> torch.Tensor:
    assert isinstance(input, torch.Tensor), "Input must be a PyTorch tensor"
    
    if out is None:
        out = torch.empty_like(input)
    
    # Convert input and output tensors to Triton-compatible types
    X = input.contiguous().view(-1).to(torch.float32)
    Y = out.contiguous().view(-1).to(torch.float32)
    
    # Get the number of elements
    N = X.shape[0]
    
    # Launch the Triton kernel
    num_warps = 4
    exp_sqrt_kernel[X, Y, N](BLOCK_SIZE=1024, num_warps=num_warps)
    
    return out
