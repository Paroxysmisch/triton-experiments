import torch
import triton
import triton.language as tl
from hardshrink import hardshrink

@triton.jit
def fused_forward(
    x_ptr, out_ptr, size,
    p, seed,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    x = tl.load(x_ptr + offset, mask=mask)
    x_keep = tl.where(tl.rand(seed, offset) > p, x / (1 - p), 0.0)
    tl.store(out_ptr + offset, hardshrink(x_keep), mask=mask)

@triton.jit
def fused_backward(
    out_grad_ptr, x_grad_ptr, size,
    p, seed,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    out_grad = tl.load(out_grad_ptr + offset, mask=mask)
    x_keep_grad = tl.where(tl.rand(seed, offset) > p, out_grad / (1 - p), 0.0)
    tl.store(x_grad_ptr + offset, hardshrink.grad(x_keep_grad), mask=mask)

def fused_hardshrink_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> torch.Tensor:
    assert (
        lambd == 0.5
    ), "Fused hardshrink dropout only supports lambda == 0.5, please use nn.functional.hardshrink instead"
    assert input.is_contiguous()

    if not training or p == 0.0:
        return input

    output = torch.empty_like(input) if not inplace else input

    size = input.numel()
    BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(size)))
    grid = lambda meta: (triton.cdiv(size, BLOCK_SIZE), )

    with torch.cuda.device(input.device.index):
        fused_forward[grid](input, output, size, p, time_seed(), BLOCK_SIZE=BLOCK_SIZE)

    return output

def fused_hardshrink_dropout_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    lambd: float = 0.5,
) -> torch.Tensor:
    assert (
        lambd == 0.5
    ), "Fused hardshrink dropout only supports lambda == 0.5, please use nn.functional.hardshrink instead"
    assert grad_output.is_contiguous()

    if not training or p == 0.0:
        return grad_output

    if not inplace:
        input = input.clone()

    size = grad_output.numel()
    BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(size)))
    grid = lambda meta: (triton.cdiv(size, BLOCK_SIZE), )

    with torch.cuda.device(grad_output.device.index):
        fused_backward[grid](
            grad_output, input, size, p, time_seed(), BLOCK_SIZE=BLOCK_SIZE)

    return input
