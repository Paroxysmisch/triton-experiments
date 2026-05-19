import torch
import triton
import triton.language as tl

# Load the Triton kernel
argmax_kernel = triton.compile(argmax_kernel, config={"block_size": 256})

def argmax(input, dim=None, keepdim=False):
    if dim is None:
        input = input.flatten()
        dim = 0
    
    n_elements = input.numel()
    stride_x = input.stride(0)
    stride_out = 1 if keepdim else 0
    
    # Allocate output tensor
    out = torch.empty(n_elements // stride_out, dtype=torch.long, device=input.device)
    
    # Launch Triton kernel
    grid_size = (n_elements + 255) // 256
    argmax_kernel[grid_size](input.data_ptr(), out.data_ptr(), n_elements, stride_x, stride_out)
    
    return out
