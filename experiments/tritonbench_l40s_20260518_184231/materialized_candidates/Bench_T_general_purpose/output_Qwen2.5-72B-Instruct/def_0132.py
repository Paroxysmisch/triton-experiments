import triton
import triton.language as tl

@triton.jit
def mul_sub_kernel(
    input_ptr, other_mul_ptr, other_sub_ptr, out_ptr,
    input_n_elements, other_mul_n_elements, other_sub_n_elements,
    alpha, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_n_elements

    input_x = tl.load(input_ptr + offsets, mask=mask)
    other_mul_x = tl.load(other_mul_ptr + offsets, mask=mask) if other_mul_n_elements > 1 else other_mul_ptr[0]
    other_sub_x = tl.load(other_sub_ptr + offsets, mask=mask) if other_sub_n_elements > 1 else other_sub_ptr[0]

    result = (input_x * other_mul_x) - (alpha * other_sub_x)
    tl.store(out_ptr + offsets, result, mask=mask)

import torch
import triton
import triton.language as tl

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> torch.Tensor:
    # Determine the number of elements in each tensor
    input_n_elements = input.numel()
    other_mul_n_elements = other_mul.numel() if isinstance(other_mul, torch.Tensor) else 1
    other_sub_n_elements = other_sub.numel() if isinstance(other_sub, torch.Tensor) else 1

    # Ensure other_mul and other_sub are tensors
    other_mul = other_mul if isinstance(other_mul, torch.Tensor) else torch.tensor(other_mul, device=input.device, dtype=input.dtype)
    other_sub = other_sub if isinstance(other_sub, torch.Tensor) else torch.tensor(other_sub, device=input.device, dtype=input.dtype)

    # Ensure other_mul and other_sub are broadcastable to the input tensor
    other_mul = other_mul.expand_as(input)
    other_sub = other_sub.expand_as(input)

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Define the grid and block size
    BLOCK_SIZE = 1024
    grid = (input_n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    mul_sub_kernel[grid](
        input, other_mul, other_sub, out,
        input_n_elements, other_mul_n_elements, other_sub_n_elements,
        alpha, BLOCK_SIZE
    )

    return out
