import triton
import triton.language as tl

@triton.jit
def mul_sub_kernel(
    input_ptr, other_mul_ptr, other_sub_ptr, alpha, out_ptr,
    n_elements, input_stride, other_mul_stride, other_sub_stride, out_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    input_data = tl.load(input_ptr + offsets * input_stride, mask=mask)
    other_mul_data = tl.load(other_mul_ptr + offsets * other_mul_stride, mask=mask)
    other_sub_data = tl.load(other_sub_ptr + offsets * other_sub_stride, mask=mask)

    # Compute the result
    result = (input_data * other_mul_data) - (alpha * other_sub_data)

    # Store the result
    tl.store(out_ptr + offsets * out_stride, result, mask=mask)

import torch

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    # Convert scalars to tensors
    if isinstance(other_mul, (int, float)):
        other_mul = torch.tensor(other_mul, dtype=input.dtype, device=input.device)
    if isinstance(other_sub, (int, float)):
        other_sub = torch.tensor(other_sub, dtype=input.dtype, device=input.device)
    
    # Broadcasting to match input shapes
    other_mul = other_mul.expand_as(input)
    other_sub = other_sub.expand_as(input)
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Get number of elements
    n_elements = input.numel()
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Define an appropriate block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    mul_sub_kernel[grid](
        input_ptr=input,
        other_mul_ptr=other_mul,
        other_sub_ptr=other_sub,
        alpha=alpha,
        out_ptr=out,
        n_elements=n_elements,
        input_stride=input.stride(0),
        other_mul_stride=other_mul.stride(0),
        other_sub_stride=other_sub.stride(0),
        out_stride=out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
