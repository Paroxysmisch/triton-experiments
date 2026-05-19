import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def add_mean_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    mean_ptr,
    n_elements,
    alpha: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_ptrs = input_ptr + offsets
    other_ptrs = other_ptr + (offsets % D)

    inputs = tl.load(input_ptrs, mask=mask)
    others = tl.load(other_ptrs, mask=mask)

    result = inputs + alpha * others
    mean_val = tl.sum(result) / n_elements
    output = mean_val

    if pid == 0:
        tl.store(output_ptr, output)
        tl.store(mean_ptr, mean_val)


def add_mean(
    input: torch.Tensor,
    other: torch.Tensor,
    dim: Optional[int] = None,
    alpha: float = 1,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if dim is not None and dim < -input.ndim or dim >= input.ndim:
        raise IndexError("Dimension out of range (expected to be in range of [{}, {}], but got {})".format(-input.ndim, input.ndim - 1, dim))

    if dim is not None:
        input = torch.movedim(input, dim, -1)
        N = int(input.numel() / input.size(-1))
        D = 1
    else:
        N = 1
        D = int(input.numel())

    if dtype is not None:
        input = input.to(dtype)

    if out is not None:
        if out.ndim != 0:
            raise RuntimeError("out must be a 0-dimensional tensor")
    else:
        out = torch.empty((), dtype=input.dtype, device=input.device)

    mean = torch.empty((), dtype=input.dtype, device=input.device)
    grid = (1,)
    add_mean_kernel[grid](
        input,
        other,
        out,
        mean,
        N * D,
        alpha,
        D=D,
        BLOCK_SIZE=1
    )

    if dim is not None:
        out = torch.movedim(out, 0, dim)
        if not keepdim:
            out = out.squeeze(dim)

    return out, mean
