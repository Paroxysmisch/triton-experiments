import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr, alpha, 
    input_stride, other_stride, output_stride, 
    input_size, other_size, output_size, 
    dim, keepdim, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    input_offsets = offsets * input_stride
    other_offsets = offsets * other_stride
    output_offsets = offsets * output_stride

    input_val = tl.load(input_ptr + input_offsets, mask=mask)
    other_val = tl.load(other_ptr + other_offsets, mask=mask)

    result = input_val + alpha * other_val

    if dim is not None:
        # Compute mean along the specified dimension
        if keepdim:
            tl.atomic_add(output_ptr + output_offsets, result, mask=mask)
        else:
            tl.atomic_add(output_ptr + (output_offsets // output_size), result, mask=mask)
    else:
        # Compute mean over all elements
        tl.atomic_add(output_ptr, result, mask=mask)

import torch
import triton
import triton.language as tl

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    # Handle dtype promotion
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    # Handle broadcasting
    input, other = torch.broadcast_tensors(input, other)
    
    # Compute the shape and strides
    input_shape = input.shape
    other_shape = other.shape
    output_shape = input_shape if dim is None else tuple(s if i != dim or keepdim else 1 for i, s in enumerate(input_shape))
    
    # Allocate output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output tensor shape mismatch"
    
    # Compute strides
    input_stride = input.stride()
    other_stride = other.stride()
    output_stride = out.stride()
    
    # Compute sizes
    input_size = input.numel()
    other_size = other.numel()
    output_size = out.numel()
    
    # Launch the Triton kernel
    grid = (triton.cdiv(output_size, 1024),)
    add_mean_kernel[grid](
        input.data_ptr(), other.data_ptr(), out.data_ptr(), alpha,
        input_stride, other_stride, output_stride,
        input_size, other_size, output_size,
        dim, keepdim,
        BLOCK_SIZE=1024
    )
    
    # Compute the mean
    if dim is not None:
        if keepdim:
            out = out / input.shape[dim]
        else:
            out = out / input.shape[dim].item()
    else:
        out = out / input.numel()
    
    return out

# Test the function
input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
other = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
alpha = 2.0
dim = 1
keepdim = True
dtype = torch.float32

result = add_mean(input, other, dim=dim, alpha=alpha, keepdim=keepdim, dtype=dtype)
print(result)
