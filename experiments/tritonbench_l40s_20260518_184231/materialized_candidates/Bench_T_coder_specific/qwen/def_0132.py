import triton
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=8),
    ],
    key=['n_elements']
)
def mul_sub(
    input: torch.Tensor,
    other_mul: torch.Tensor,
    other_sub: torch.Tensor,
    alpha: float = 1.0,
    out: torch.Tensor = None,
) -> torch.Tensor:
    assert input.dim() == 1, "Input must be a 1D tensor"
    assert other_mul.dim() == 1, "Other multiply tensor must be a 1D tensor"
    assert other_sub.dim() == 1, "Other subtract tensor must be a 1D tensor"
    assert input.size(0) == other_mul.size(0), "Input and other multiply tensors must have the same size"
    assert input.size(0) == other_sub.size(0), "Input and other subtract tensors must have the same size"

    n_elements = input.size(0)

    if out is None:
        out = torch.empty_like(input)

    # Ensure the output tensor has the correct shape
    assert out.size(0) == n_elements, "Output tensor must have the same size as input tensor"

    grid_size = (n_elements + 255) // 256

    mul_sub_kernel[grid_size, 256](input, other_mul, other_sub, out, n_elements, alpha)

    return out
