import triton
import triton.language as tl

@triton.jit
def asin_kernel(
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
    
    # Compute arcsine
    output_vec = tl.asin(input_vec)
    
    # Store the result
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def asin(input, *, out=None):
    # Ensure input is a Torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Ensure input is a float type
    if input.dtype not in [torch.float16, torch.float32, torch.float64]:
        raise TypeError("Input tensor must be of float type")
    
    # Ensure input is in the range [-1, 1]
    if not torch.all((input >= -1) & (input <= 1)):
        raise ValueError("Input tensor must be in the range [-1, 1]")
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as the input tensor")
        if out.dtype != input.dtype:
            raise TypeError("Output tensor must have the same dtype as the input tensor")
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    asin_kernel[grid](
        input.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        n_elements,
        BLOCK_SIZE=1024
    )
    
    return out
