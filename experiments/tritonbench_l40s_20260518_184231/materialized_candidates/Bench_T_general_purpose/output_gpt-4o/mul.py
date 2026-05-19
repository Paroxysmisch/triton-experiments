import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    input_ptr, other_ptr, out_ptr,
    input_stride, other_stride, out_stride,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Create a block of indices
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offsets < N
    
    # Load data from memory
    input_data = tl.load(input_ptr + offsets * input_stride, mask=mask)
    other_data = tl.load(other_ptr + offsets * other_stride, mask=mask)
    
    # Perform element-wise multiplication
    result = input_data * other_data
    
    # Store the result back to the output
    tl.store(out_ptr + offsets * out_stride, result, mask=mask)

import torch

def mul(input, other, *, out=None):
    # Ensure inputs are tensors
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    # Broadcast input and other to a common shape
    input, other = torch.broadcast_tensors(input, other)
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Get the number of elements
    N = input.numel()
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Example block size, can be tuned
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    mul_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        out_ptr=out.data_ptr(),
        input_stride=input.stride(0),
        other_stride=other.stride(0),
        out_stride=out.stride(0),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
