import triton
import triton.language as tl

@triton.jit
def reciprocal_kernel(
    input_ptr,  # *Pointer* to the input tensor
    output_ptr,  # *Pointer* to the output tensor
    n_elements,  # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the reciprocal
    output_vec = 1.0 / input_vec
    
    # Store the result
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def reciprocal(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a tensor")
    
    # Promote integral types to the default scalar type (float32)
    if input.dtype in [torch.int32, torch.int64]:
        input = input.to(torch.float32)
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as the input tensor")
        if out.dtype != torch.float32:
            raise TypeError("Output tensor must be of type float32")
    
    # Launch the Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    reciprocal_kernel[grid](
        input.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        input.numel(),
        BLOCK_SIZE=1024
    )
    
    return out
