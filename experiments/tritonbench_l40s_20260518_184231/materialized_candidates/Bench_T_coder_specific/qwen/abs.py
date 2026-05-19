import triton
import torch

@torch.jit.script
def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure the input and output tensors are on the same device
    input = input.to(device=out.device)
    out = out.to(device=input.device)
    
    # Launch the Triton kernel
    grid_size = (triton.cdiv(input.numel(), 256),)
    abs_kernel[input.numel()](input.data_ptr(), out.data_ptr(), input.numel())
    
    return out
