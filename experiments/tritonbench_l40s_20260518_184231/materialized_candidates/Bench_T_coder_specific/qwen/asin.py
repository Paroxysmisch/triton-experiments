import triton
import torch

@torch.jit.script
def asin(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    asin_kernel[triton.cdiv(n_elements, 256), 256](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
