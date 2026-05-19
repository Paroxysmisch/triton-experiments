import triton
import triton.language as tl

@triton.jit
def gelu_kernel(X, Y, N, approximate, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    if approximate == 0:  # 'none'
        y = x * tl.erf(x / tl.sqrt(tl.float32(2.0))) * tl.float32(0.5) + x * tl.float32(0.5)
    else:  # 'tanh'
        cdf = tl.tanh(tl.sqrt(tl.float32(2.0) / tl.float32(3.141592653589793)) * (x + tl.float32(0.044715) * x * x * x))
        y = x * (tl.float32(1.0) + cdf) * tl.float32(0.5)
    
    tl.store(Y + offsets, y, mask=mask)

@triton.jit
def std_kernel(Y, STD, N, dim, keepdim, correction, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    y = tl.load(Y + offsets, mask=mask)
    
    mean = tl.sum(y, axis=0) / N
    var = tl.sum((y - mean) ** 2, axis=0) / (N - correction)
    std = tl.sqrt(var)
    
    if keepdim:
        tl.store(STD + offsets, std, mask=mask)
    else:
        tl.store(STD + pid, std, mask=mask)

import torch
import triton
import triton.language as tl

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Convert approximate string to integer for kernel
    approximate = 0 if approximate == 'none' else 1
    
    # Determine the shape and size of the input tensor
    input_shape = input.shape
    input_size = input.numel()
    
    # Allocate output tensor for GELU result
    gelu_output = torch.empty_like(input)
    
    # Launch GELU kernel
    grid = (input_size // 1024 + 1,)
    gelu_kernel[grid](input, gelu_output, input_size, approximate, BLOCK_SIZE=1024)
    
    # Determine the dimensions for standard deviation calculation
    if dim is None:
        dim = tuple(range(len(input_shape)))
    elif isinstance(dim, int):
        dim = (dim,)
    
    # Calculate the number of elements in the reduced dimensions
    reduced_size = 1
    for d in dim:
        reduced_size *= input_shape[d]
    
    # Allocate output tensor for standard deviation
    if out is None:
        if keepdim:
            out_shape = tuple(input_shape[d] if d not in dim else 1 for d in range(len(input_shape)))
        else:
            out_shape = tuple(input_shape[d] for d in range(len(input_shape)) if d not in dim)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    
    # Launch standard deviation kernel
    grid = (reduced_size // 1024 + 1,)
    std_kernel[grid](gelu_output, out, reduced_size, dim, keepdim, correction, BLOCK_SIZE=1024)
    
    return out

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None) -> Tensor:
    # Function body as provided above
