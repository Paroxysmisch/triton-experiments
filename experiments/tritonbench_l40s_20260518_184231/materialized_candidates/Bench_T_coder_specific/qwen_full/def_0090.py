import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardshrink_dropout_kernel(
    input_pointer, output_pointer, n_elements, p, lambd, training, inplace,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(input_pointer + offsets, mask=mask)

    if inplace:
        out = x
    else:
        out = tl.where(mask, x, 0.0)

    if training:
        rand = tl.rand(offsets)
        out = tl.where(rand > p, out, 0.0)

    s = tl.abs(out) - lambd
    out = tl.where(s <= 0, 0.0, out)

    if inplace:
        tl.store(input_pointer + offsets, out, mask=mask)
    else:
        tl.store(output_pointer + offsets, out, mask=mask)

def fused_hardshrink_dropout(
    input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5
) -> torch.Tensor:
    assert input.is_contiguous()

    assert 0.0 <= p <= 1.0, f"p must be in [0.0, 1.0], but got {p}"
    assert 0.0 <= lambd, f"lambd must be in [0.0, +inf), but got {lambd}"

    output = input if inplace else torch.empty_like(input)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    fused_hardshrink_dropout_kernel[grid](input, output, n_elements, p, lambd, training, inplace, BLOCK_SIZE=1024)

    return output
