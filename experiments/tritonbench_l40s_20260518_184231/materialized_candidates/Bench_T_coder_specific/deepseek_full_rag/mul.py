import torch
import triton
import triton.language as tl
from triton import next_power_of_2
from triton.language.core import load, store
from torch._inductor.triton_heuristics import group_by_cu_block_and_warp
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def mul_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    program_descriptor = instance_descriptor(n_elements, BLOCK_M, BLOCK_N)
    pid_m, pid_n = group_by_cu_block_and_warp(program_descriptor, pid)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offs = offs_m * n_elements + offs_n
    mask = offs < n_elements
    input = load(input_ptr + offs, mask=mask)
    other = load(other_ptr + offs, mask=mask)
    result = input * other
    store(output_ptr + offs, result, mask=mask)

def mul(
    input: torch.Tensor,
    other: torch.Tensor | float | int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_contiguous()
    assert (
        isinstance(other, (float, int, torch.Tensor))
        or (isinstance(other, torch.Tensor) and other.is_contiguous())
    ), f"unsupported types {type(other)}"
    n_elements = out.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_M"] * meta["BLOCK_N"]),
    )
    mul_kernel[grid](input, other, out, n_elements, BLOCK_M=128, BLOCK_N=128)
    return out
