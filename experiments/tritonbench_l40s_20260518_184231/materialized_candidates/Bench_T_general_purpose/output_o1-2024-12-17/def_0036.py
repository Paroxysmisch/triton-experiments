import triton
import triton.language as tl

@triton.jit
def _add_gelu_kernel(
    input_ptr, other_ptr, out_ptr,
    alpha, N, approximate, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < N

    inp = tl.load(input_ptr + block_start, mask=mask, other=0.0)
    oth = tl.load(other_ptr + block_start, mask=mask, other=0.0)
    val = inp + alpha * oth

    if approximate == 0:
        # GELU exact: x * 0.5 * (1 + erf(x / sqrt(2)))
        val = 0.5 * val * (1.0 + tl.erf(val / tl.sqrt(2.0)))
    else:
        # GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        val = 0.5 * val * (
            1.0 + tl.tanh(
                0.7978845608028654 * (val + 0.044715 * val * val * val)
            )
        )

    tl.store(out_ptr + block_start, val, mask=mask)


def add_gelu(input, other, alpha=1, approximate='none', out=None):
    import torch

    if not torch.is_tensor(input):
        raise TypeError("input must be a torch.Tensor")
    if not (torch.is_tensor(other) or isinstance(other, (float, int))):
        raise TypeError("other must be a torch.Tensor or a number")

    if out is None:
        out = torch.empty_like(input)

    # Ensure same device and type
    if torch.is_tensor(other):
        other_data = other.to(input.device).flatten()
    else:
        other_data = torch.full_like(input, float(other)).flatten()

    input_data = input.flatten()
    out_data = out.flatten()

    N = input_data.numel()
    # Convert approximate to int (0 - exact, 1 - tanh)
    approximate_flag = 0 if approximate == 'none' else 1

    grid = lambda meta: ( (N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _add_gelu_kernel[grid](
        input_data, other_data, out_data,
        alpha, N, approximate_flag,
        BLOCK_SIZE=1024
    )
    return out.view(input.shape)
