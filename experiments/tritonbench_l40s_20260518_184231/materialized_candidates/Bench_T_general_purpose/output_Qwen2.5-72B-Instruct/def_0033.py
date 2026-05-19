import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    dim,  # Dimension to reduce
    stride,  # Stride along the dimension to reduce
    num_elements,  # Total number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute the max value in the block
    max_val = tl.max(x, axis=0)
    
    # Compute the exponentials
    x_exp = tl.exp(x - max_val)
    
    # Compute the sum of exponentials
    sum_exp = tl.sum(x_exp, axis=0)
    
    # Compute the log of the sum of exponentials
    result = tl.log(sum_exp) + max_val
    
    # Store the result
    tl.store(output_ptr + pid, result)

import torch
import triton
import triton.language as tl

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    # Check input and dimension
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if dim < 0 or dim >= input.dim():
        raise ValueError("dim must be within the range of input dimensions")
    
    # Determine the output shape
    output_shape = list(input.shape)
    if not keepdim:
        output_shape.pop(dim)
    else:
        output_shape[dim] = 1
    
    # Allocate output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != output_shape:
            raise ValueError(f"out tensor shape {out.shape} does not match the expected shape {output_shape}")
    
    # Compute the number of elements to reduce and the stride
    num_elements = input.shape[dim]
    stride = input.stride(dim)
    
    # Launch the Triton kernel
    grid = (out.numel(),)
    logsumexp_kernel[grid](
        input.contiguous().data_ptr(),
        out.data_ptr(),
        dim,
        stride,
        num_elements,
        BLOCK_SIZE=1024
    )
    
    return out
