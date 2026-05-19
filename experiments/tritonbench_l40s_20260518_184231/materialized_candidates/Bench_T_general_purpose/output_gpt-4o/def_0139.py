import triton
import triton.language as tl

@triton.jit
def variance_kernel(X_ptr, mean_ptr, out_ptr, N, stride, correction, BLOCK_SIZE: tl.constexpr):
    # Calculate the variance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    X = tl.load(X_ptr + offsets * stride, mask=offsets < N, other=0.0)
    mean = tl.load(mean_ptr)
    
    # Compute squared differences
    diff = X - mean
    sq_diff = diff * diff
    
    # Sum squared differences
    var_sum = tl.sum(sq_diff, axis=0)
    
    # Apply correction and compute variance
    variance = var_sum / tl.max(0, N - correction)
    
    # Store result
    tl.store(out_ptr + pid, variance)

@triton.jit
def mean_kernel(X_ptr, out_ptr, N, stride, BLOCK_SIZE: tl.constexpr):
    # Calculate the mean
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    X = tl.load(X_ptr + offsets * stride, mask=offsets < N, other=0.0)
    
    # Sum and compute mean
    sum_X = tl.sum(X, axis=0)
    mean = sum_X / N
    
    # Store result
    tl.store(out_ptr + pid, mean)

import torch

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    # Handle the dimension
    if dim is None:
        # Reduce over all dimensions
        input = input.flatten()
        dim = 0
    elif isinstance(dim, int):
        dim = (dim,)
    
    # Calculate the mean
    N = input.shape[dim[0]]
    mean = torch.empty(1, device=input.device, dtype=input.dtype)
    mean_kernel[(1,)](input, mean, N, input.stride(dim[0]), BLOCK_SIZE=1024)
    
    # Calculate the variance
    variance = torch.empty(1, device=input.device, dtype=input.dtype)
    variance_kernel[(1,)](input, mean, variance, N, input.stride(dim[0]), correction, BLOCK_SIZE=1024)
    
    # Calculate the standard deviation
    std_dev = torch.sqrt(variance)
    
    # Prepare the output
    if keepdim:
        std_dev = std_dev.view(*[1 if i in dim else s for i, s in enumerate(input.shape)])
    
    if out is not None:
        out.copy_(std_dev)
        return out
    else:
        return std_dev

# Example usage
x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
result = std(x, correction=1, keepdim=True)
print(result)
