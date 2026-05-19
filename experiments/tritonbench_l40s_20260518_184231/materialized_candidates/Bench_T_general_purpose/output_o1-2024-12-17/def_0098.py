import math
import torch
import triton
import triton.language as tl

@triton.jit
def _sub_gelu_kernel(
    input_ptr, other_ptr, out_ptr,
    alpha, approx_mode,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(input_ptr + offsets, mask=mask)
    oth = tl.load(other_ptr + offsets, mask=mask)

    x = inp - alpha * oth

    # approximate: 0 => exact, 1 => tanh
    if approx_mode == 0:
        # exact gelu
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        val = 0.5 * x * (1.0 + tl.erf(x / sqrt_2))
    else:
        # tanh gelu
        # 0.5 * x * (1 + tanh( sqrt(2/pi)*( x + 0.044715*x^3 ) ))
        sqrt_2_pi = (2.0 / math.pi)**0.5
        x_cubed = x * x * x
        inner = sqrt_2_pi * (x + 0.044715 * x_cubed)
        val = 0.5 * x * (1.0 + tl.tanh(inner))

    tl.store(out_ptr + offsets, val, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    """
    def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> Tensor:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to subtract from input.
        alpha (Number, optional): The multiplier for other. Default is 1.
        approximate (str, optional): The approximation method for GELU. Default is 'none'.
        out (Tensor, optional): The output tensor.
    Math:
        out_i = GELU(input_i - alpha * other_i)
    GELU(x) = x * Φ(x) when approximate is 'none'
    GELU(x) = 0.5 * x * (1 + Tanh(√(2/π)*(x + 0.044715*x^3))) when approximate is 'tanh'
    """
    if not torch.is_tensor(input):
        raise TypeError("input must be a torch.Tensor")
    if not torch.is_tensor(other):
        if not isinstance(other, (int, float)):
            raise TypeError("other must be a torch.Tensor or a number")
        other = torch.tensor(other, dtype=input.dtype, device=input.device)

    if out is None:
        out = torch.empty_like(input)

    # Broadcast logic
    input_b, other_b = torch.broadcast_tensors(input, other)

    n_elements = input_b.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    approx_mode = 0 if approximate == 'none' else 1

    _sub_gelu_kernel[grid](
        input_b, other_b, out,
        alpha, approx_mode,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
