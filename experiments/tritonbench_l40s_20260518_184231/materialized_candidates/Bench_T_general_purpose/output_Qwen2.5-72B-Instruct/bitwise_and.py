import triton
import triton.language as tl

@triton.jit
def bitwise_and_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor
    output_ptr, # Pointer to the output tensor
    n_elements, # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vec = tl.load(input_ptr + offsets, mask=mask)
    other_vec = tl.load(other_ptr + offsets, mask=mask)

    output_vec = input_vec & other_vec

    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def bitwise_and(input, other, *, out=None):
    # Ensure input and other are of the same type and are integral or Boolean
    if input.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool] or \
       other.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]:
        raise ValueError("Input tensors must be of integral or Boolean types.")

    if input.dtype != other.dtype:
        raise ValueError("Input tensors must have the same data type.")

    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()

    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the kernel
    bitwise_and_kernel[grid](
        input.contiguous().view(-1).cuda(),
        other.contiguous().view(-1).cuda(),
        out.contiguous().view(-1).cuda(),
        n_elements,
        BLOCK_SIZE
    )

    return out
