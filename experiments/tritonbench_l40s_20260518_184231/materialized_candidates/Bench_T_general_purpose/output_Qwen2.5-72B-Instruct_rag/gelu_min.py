import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, sqrt, pow

# Triton kernel for GELU using the exact method
@triton.jit
def gelu_none_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    tl.store(Y + offsets, x_gelu, mask=mask)

# Triton kernel for GELU using the tanh approximation
@triton.jit
def gelu_tanh_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + tanh(sqrt(2 / 3.141592653589793) * (x_fp32 + 0.044715 * pow(x_fp32, 3))))
    tl.store(Y + offsets, x_gelu, mask=mask)

# Wrapper function for applying GELU and computing the minimum
def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Determine the shape of the output tensor
    if dim is None:
        output_shape = (1,)
    else:
        output_shape = list(input.shape)
        if not keepdim:
            output_shape[dim] = 1

    # Allocate the output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Apply GELU using the specified method
    if approximate == 'none':
        gelu_kernel = gelu_none_kernel
    elif approximate == 'tanh':
        gelu_kernel = gelu_tanh_kernel
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    # Launch the GELU kernel
    BLOCK_SIZE = 256
    grid = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    gelu_kernel[grid](input, out, BLOCK_SIZE)

    # Compute the minimum value
    if dim is None:
        min_value, min_index = torch.min(out, out=out)
        return min_value
    else:
        min_value, min_index = torch.min(out, dim=dim, keepdim=keepdim)
        return min_value, min_index

# Example usage
input_tensor = torch.randn(4, 5, device='cuda')
result = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)
print(result)
