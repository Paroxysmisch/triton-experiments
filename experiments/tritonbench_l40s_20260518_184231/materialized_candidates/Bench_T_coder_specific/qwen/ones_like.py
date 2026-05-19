import torch

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Determine the shape, dtype, and device of the input tensor
    shape = input.shape
    dtype = dtype or input.dtype
    device = device or input.device
    
    # Allocate the output tensor
    output = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad, layout=layout, memory_format=memory_format)
    
    # Get the total number of elements
    N = output.numel()
    
    # Configure the grid and block sizes
    BLOCK_SIZE = 256
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    ones_like_kernel[grid=(num_blocks,), block=(BLOCK_SIZE,)](output.data_ptr(), N, BLOCK_SIZE)
    
    return output
