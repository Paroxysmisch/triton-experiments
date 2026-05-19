import triton
import triton.language as tl
import torch

# Constants for the SELU function
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

# Triton kernel for element-wise SELU
@triton.jit
def _selu_kernel(X, OUT, N, inplace: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * N + tl.arange(0, N)
    x = tl.load(X + offsets)
    
    # Apply SELU transformation
    selu_x = SCALE * (tl.maximum(0, x) + tl.minimum(0, ALPHA * (tl.exp(x) - 1)))
    
    # Store result
    if inplace:
        tl.store(X + offsets, selu_x)
    else:
        tl.store(OUT + offsets, selu_x)

# Wrapper function for the SELU kernel
def selu(input, inplace=False):
    assert input.is_cuda, "Input tensor must be a CUDA tensor"
    N = input.numel()
    
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _selu_kernel[grid](input, out, N, inplace=inplace, BLOCK_SIZE=1024)
    
    return out

# Example usage
x = torch.randn(1024, device='cuda')
y = selu(x)
