import math
import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(
    in_ptr, 
    out_ptr, 
    n, 
    approximate_tanh: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    if approximate_tanh:
        # tanh approximation
        # 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        y = x + 0.044715 * (x ** 3)
        out_val = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * y))
    else:
        # exact version
        # x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        out_val = x * 0.5 * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offsets, out_val, mask=mask)


def gelu(input: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise:
      - If approximate='none': GELU(x) = x * Φ(x)
      - If approximate='tanh': GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    assert approximate in ['none', 'tanh'], "approximate must be 'none' or 'tanh'"

    # Ensure input is a CUDA tensor
    assert input.is_cuda, "Input tensor must be on CUDA device"

    output = torch.empty_like(input)
    n = input.numel()

    # Launch the kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    triton.run(
        _gelu_kernel,
        grid=grid,
        args=[
            input.data_ptr(),
            output.data_ptr(),
            n,
            (approximate == 'tanh')
        ],
        num_warps=4,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
