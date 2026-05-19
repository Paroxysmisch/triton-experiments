import triton
import triton.language as tl

@triton.jit
def ones_like_kernel(input_ptr, output_ptr, size, dtype):
    # Fill the output tensor with ones
    for i in range(size):
        output_ptr[i] = tl.ones(dtype=dtype)

import torch

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Determine the size of the input tensor
    size = input.numel()
    
    # Set default values for dtype, layout, and device if not provided
    if dtype is None:
        dtype = input.dtype
    if layout is None:
        layout = input.layout
    if device is None:
        device = input.device

    # Create an output tensor
    output = torch.empty(size, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

    # Launch the Triton kernel
    ones_like_kernel[(1,)](input_ptr=input.data_ptr(), output_ptr=output.data_ptr(), size=size, dtype=dtype)

    return output
