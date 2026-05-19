import torch
import triton
import triton.language as tl

device = 'cuda:0'

def rsqrt(input: torch.Tensor, *, out=None) -> torch.Tensor:
    # Get the shape of the input tensor
    n = input.numel()
    
    # Prepare an output tensor on the same device
    if out is None:
        output = torch.empty_like(input).to(device)
    else:
        output = out.to(device)
    
    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and output.is_cuda, 'Input or Output tensor is not on GPU'
    
    # Launch the Triton kernel
    rsqrt_kernel[(n // 32,) if n > 32 else (1,)](X_ptr=input, Y_ptr=output, N=n)
    
    return output
