import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow

@triton.jit
def _sub_gelu_none_kernel(
    x_ptr, other_ptr,
    out_ptr,
    alpha_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)

    # Subtraction: (input - alpha * other)
    val = x - alpha * y

    # Exact GELU:  x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # Using 1/sqrt(2) ~ 0.7071067811
    gelu_val = 0.5 * val * (1.0 + erf(0.7071067811 * val))

    tl.store(out_ptr + offsets, gelu_val, mask=mask)


@triton.jit
def _sub_gelu_tanh_kernel(
    x_ptr, other_ptr,
    out_ptr,
    alpha_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)

    # Subtraction: (input - alpha * other)
    val = x - alpha * y

    # Approximate GELU (tanh):
    # 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
    # sqrt(2/pi) ~ 0.7978845608
    c0 = 0.7978845608
    c1 = 0.044715
    gelu_val = 0.5 * val * (
        1.0 + tanh(c0 * (val + c1 * pow(val, 3)))
    )

    tl.store(out_ptr + offsets, gelu_val, mask=mask)


def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    """
    Subtracts 'other', scaled by 'alpha', from 'input' and then applies GELU to the result.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to subtract from input.
        alpha (Number, optional): The multiplier for 'other'. Default is 1.
        approximate (str, optional): The approximation method for GELU. 'none' for exact, 'tanh' for approximate.
        out (Tensor, optional): The output tensor.
    
    Returns:
        Tensor: The result of applying the GELU function to (input - alpha * other).
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch Tensor")
    
    # Handle 'other' if it is a scalar
    if isinstance(other, (float, int)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast 'other' to match 'input'
    broadcasted_other = torch.broadcast_to(other, input.shape)

    # Convert alpha to tensor so it can be loaded in Triton
    alpha_tensor = torch.tensor([alpha], dtype=input.dtype, device=input.device)

    # Allocate output if needed
    if out is None:
        out = torch.empty_like(input)

    # Flatten all arrays for simplicity
    x_flat = input.contiguous().view(-1)
    y_flat = broadcasted_other.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)
    n_elements = x_flat.numel()

    # Grid setting
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Choose kernel based on approximation mode
    if approximate == 'none':
        _sub_gelu_none_kernel[grid](
            x_flat, y_flat,
            out_flat,
            alpha_tensor,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    elif approximate == 'tanh':
        _sub_gelu_tanh_kernel[grid](
            x_flat, y_flat,
            out_flat,
            alpha_tensor,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError("approximate must be either 'none' or 'tanh'")

    return out.view(input.shape)
