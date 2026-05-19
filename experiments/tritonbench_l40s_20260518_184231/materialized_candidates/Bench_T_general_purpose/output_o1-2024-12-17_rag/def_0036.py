import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh

@triton.jit
def _add_gelu_none_kernel(
    input_ptr, other_ptr, out_ptr,
    alpha, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    val = x + alpha * y
    # GELU (exact/none): 0.5 * val * (1 + erf(val / sqrt(2)))
    gelu_val = 0.5 * val * (1 + erf(val * 0.7071067811865475))
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

@triton.jit
def _add_gelu_tanh_kernel(
    input_ptr, other_ptr, out_ptr,
    alpha, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    val = x + alpha * y
    # GELU (tanh): 0.5 * val * (1 + tanh( sqrt(2/pi)*(val + 0.044715*val^3) ))
    # sqrt(2/pi) ~ 0.79788456
    gelu_val = 0.5 * val * (1 + tanh(0.79788456 * (val + 0.044715 * val * val * val)))
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if not torch.is_tensor(input):
        raise TypeError("input must be a Tensor")
    # Ensure 'other' is a tensor
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)

    # Broadcast if needed
    if input.shape != other.shape:
        other = other.expand_as(input)

    # Prepare output
    if out is None:
        out = torch.empty_like(input)

    # Flatten for Triton
    input_flat = input.contiguous().view(-1)
    other_flat = other.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    n_elements = input_flat.numel()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    if approximate == 'none':
        _add_gelu_none_kernel[grid](
            input_flat, other_flat, out_flat,
            alpha, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    elif approximate == 'tanh':
        _add_gelu_tanh_kernel[grid](
            input_flat, other_flat, out_flat,
            alpha, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    return out.view(input.shape)
