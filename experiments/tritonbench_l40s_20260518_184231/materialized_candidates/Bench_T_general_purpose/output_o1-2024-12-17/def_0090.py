import triton
import triton.language as tl
import torch

@triton.jit
def _fused_hardshrink_dropout_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    p,
    lambd,
    training,
    inplace_val,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    keep_prob = 1.0 - p
    scale = 1.0 / keep_prob
    is_training = training > 0
    in_place = inplace_val > 0

    if is_training:
        rng = tl.random.xorshift32(seed, offsets)
        rand_vals = rng.to(tl.float32) * (1.0 / 4294967296.0)
        dropped = rand_vals < p
        x = tl.where(dropped, 0.0, x * scale)

    abs_x = tl.abs(x)
    x = tl.where(abs_x < lambd, 0.0, x)

    if in_place:
        tl.store(in_ptr + offsets, x, mask=mask)
    else:
        tl.store(out_ptr + offsets, x, mask=mask)

def fused_hardshrink_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5
) -> torch.Tensor:
    n_elements = input.numel()
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)

    seed = torch.randint(0, 2**31, (1,)).item()
    grid = ((n_elements + 1023) // 1024,)
    _fused_hardshrink_dropout_kernel[grid](
        input,
        out,
        n_elements,
        p,
        lambd,
        int(training),
        int(inplace),
        seed,
        BLOCK_SIZE=1024
    )
    return out
