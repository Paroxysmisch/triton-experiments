import triton
import torch

@torch.jit.script
def exp(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Get the size of the input tensor
    N = input.numel()
    
    # Launch the Triton kernel
    block_size = 128
    grid_size = (N + block_size - 1) // block_size
    
    exp_kernel[grid_size, block_size](input.data_ptr(), out.data_ptr(), N)
    
    return out
