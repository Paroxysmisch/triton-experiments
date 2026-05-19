import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(input_ptr, output_ptr, dim_size, stride_in, stride_out, BLOCK_SIZE: tl.constexpr):
    # Offsets for the input and output pointers
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load the input elements
    input_elements = tl.load(input_ptr + offsets * stride_in, mask=offsets < dim_size, other=0.0)
    
    # Compute exp(x)
    exp_elements = tl.exp(input_elements)
    
    # Compute sum(exp(x))
    sum_exp = tl.sum(exp_elements, axis=0)
    
    # Compute log(sum(exp(x)))
    logsumexp_result = tl.log(sum_exp)
    
    # Store the result
    tl.store(output_ptr + pid * stride_out, logsumexp_result)


import torch

def logsumexp(input, dim, keepdim=False, *, out=None):
    # Validate the input tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Determine the shape and strides
    input_shape = input.shape
    input_strides = input.stride()
    
    # Handle the output tensor
    if out is None:
        if keepdim:
            output_shape = list(input_shape)
            output_shape[dim] = 1
        else:
            output_shape = [s for i, s in enumerate(input_shape) if i != dim]
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define a suitable block size
    grid = (triton.cdiv(input_shape[dim], BLOCK_SIZE),)
    logsumexp_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        dim_size=input_shape[dim],
        stride_in=input_strides[dim],
        stride_out=out.stride()[0],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if keepdim:
        out = out.unsqueeze(dim)
    
    return out
