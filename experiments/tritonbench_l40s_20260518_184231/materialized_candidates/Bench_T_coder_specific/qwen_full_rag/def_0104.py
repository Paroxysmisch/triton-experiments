import triton
import triton.language as tl
import torch
from typing import Tuple

# Triton kernel to convert radians to degrees and calculate square root
@triton.jit
def _rad2deg_sqrt_kernel(input, output1, output2, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(input + idx)
    y = tl.rad2deg(x)
    z = tl.sqrt(x)
    tl.store(output1 + idx, y)
    tl.store(output2 + idx, z)

# Wrapper function to call the Triton kernel
def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    output1 = torch.empty_like(input)
    output2 = torch.empty_like(input)
    assert input.is_cuda and output1.is_cuda and output2.is_cuda
    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta["N"]), )
    _rad2deg_sqrt_kernel[grid](input, output1, output2, N)
    return output1, output2
