import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the reciprocal of the square-root
    output_vec = 1.0 / tl.sqrt(input_vec)
    
    # Handle negative values by setting them to 'nan'
    output_vec = tl.where(input_vec < 0, float('nan'), output_vec)
    
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def rsqrt(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Ensure input is on the same device as the Triton kernel
    device = input.device
    if device.type != 'cuda':
        raise ValueError("Input tensor must be on a CUDA device")
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("Output must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as the input tensor")
        if out.device != device:
            raise ValueError("Output tensor must be on the same device as the input tensor")
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rsqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out
