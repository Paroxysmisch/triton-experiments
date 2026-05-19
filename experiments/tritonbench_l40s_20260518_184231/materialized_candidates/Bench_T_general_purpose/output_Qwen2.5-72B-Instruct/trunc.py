import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    output_vec = tl.where(input_vec >= 0, tl.floor(input_vec), tl.ceil(input_vec))
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def trunc(input, *, out=None):
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out

    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    trunc_kernel[grid](
        input, out, n_elements, BLOCK_SIZE=1024
    )
    return out
