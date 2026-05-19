import triton
import triton.language as tl

@triton.jit
def polygamma_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n,  # Order of the polygamma function
    size,  # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    output_vec = tl.zeros_like(input_vec)

    # Compute the n-th derivative of the digamma function
    for i in range(n + 1):
        if i == 0:
            output_vec = tl.math.digamma(input_vec)
        else:
            output_vec = tl.math.polygamma(i - 1, input_vec)

    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def polygamma(n, input, *, out=None) -> torch.Tensor:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")

    if not isinstance(input, torch.Tensor):
        raise ValueError("input must be a torch.Tensor")

    if out is not None and not isinstance(out, torch.Tensor):
        raise ValueError("out must be a torch.Tensor or None")

    if out is None:
        out = torch.empty_like(input)

    # Ensure the input and output tensors are on the same device
    if input.device != out.device:
        raise ValueError("input and out must be on the same device")

    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    polygamma_kernel[grid](
        input,  # input_ptr
        out,  # output_ptr
        n,  # n
        input.numel(),  # size
        BLOCK_SIZE=1024  # BLOCK_SIZE
    )

    return out
