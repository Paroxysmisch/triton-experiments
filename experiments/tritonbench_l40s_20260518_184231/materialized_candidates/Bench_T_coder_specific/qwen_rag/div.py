import triton
import triton.language as tl
import torch
import math

# Define the Triton kernel for division
@triton.jit
def div_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Compute the index within the block
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform division with type promotion
    if tl.is_floating_point(x.dtype) and tl.is_floating_point(y.dtype):
        out = x / y
    elif tl.is_integer(x.dtype) and tl.is_integer(y.dtype):
        out = x.astype(tl.float32) / y.astype(tl.float32)
    else:
        out = x.astype(tl.float32) / y.astype(tl.float32)
    
    # Apply rounding mode if specified
    if rounding_mode == "floor":
        out = tl.floor(out)
    elif rounding_mode == "ceil":
        out = tl.ceil(out)
    elif rounding_mode == "round":
        out = tl.round(out)
    else:
        pass  # Default is no rounding
    
    # Store the result back to global memory
    tl.store(out_ptr + offsets, out, mask=mask)

# Wrapper function to dispatch the kernel
def div(input, other, *, rounding_mode=None, out=None):
    # Determine the shapes and dtypes of input and other
    input_shape = input.shape
    other_shape = other.shape
    dtype = torch.promote_types(input.dtype, other.dtype)
    
    # Broadcast shapes if necessary
    if len(input_shape) > len(other_shape):
        other_shape = (1,) * (len(input_shape) - len(other_shape)) + other_shape
    elif len(input_shape) < len(other_shape):
        input_shape = (1,) * (len(other_shape) - len(input_shape)) + input_shape
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(input_shape, dtype=dtype, device=input.device)
    
    # Ensure the output tensor has the same shape as input
    assert out.shape == input_shape, "Output tensor must have the same shape as input"
    
    # Get the total number of elements
    n_elements = input.numel()
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Dispatch the kernel
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    div_kernel[grid_size, BLOCK_SIZE](input.data_ptr(), other.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE)
    
    return out
