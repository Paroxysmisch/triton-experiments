import torch
import triton
import triton.language as tl

# Triton kernel for GELU using the exact method (error function)
@triton.jit
def gelu_none_kernel(X, Y, alpha, OUT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    x_alpha_y = x + alpha * y
    x_gelu = 0.5 * x_alpha_y * (1 + tl.math.erf(x_alpha_y / 1.41421356237))
    tl.store(OUT + offsets, x_gelu, mask=mask)

# Triton kernel for GELU using the tanh approximation
@triton.jit
def gelu_tanh_kernel(X, Y, alpha, OUT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    x_alpha_y = x + alpha * y
    x_gelu = 0.5 * x_alpha_y * (1 + tl.math.tanh(0.79788456 * x_alpha_y * (1 + 0.044715 * x_alpha_y * x_alpha_y)))
    tl.store(OUT + offsets, x_gelu, mask=mask)

# Wrapper function for the add_gelu operation
def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)
    
    if approximate == 'none':
        grid = (triton.cdiv(input.numel(), 1024),)
        gelu_none_kernel[grid](input, other, alpha, out, BLOCK_SIZE=1024)
    elif approximate == 'tanh':
        grid = (triton.cdiv(input.numel(), 1024),)
        gelu_tanh_kernel[grid](input, other, alpha, out, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")
    
    return out
