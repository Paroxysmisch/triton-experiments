import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor or number
    out_ptr,    # Pointer to the output tensor
    input_shape,  # Shape of the input tensor
    other_shape,  # Shape of the other tensor
    output_shape, # Shape of the output tensor
    input_strides,  # Strides of the input tensor
    other_strides,  # Strides of the other tensor
    output_strides, # Strides of the output tensor
    input_dtype: tl.dtype,  # Data type of the input tensor
    other_dtype: tl.dtype,  # Data type of the other tensor
    output_dtype: tl.dtype,  # Data type of the output tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the linear index for each element in the output tensor
    output_linear_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for dim in range(len(output_shape)):
        output_linear_index += (offsets % output_shape[dim]) * output_strides[dim]
        offsets //= output_shape[dim]

    # Load the input and other tensors
    input_linear_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    other_linear_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for dim in range(len(output_shape)):
        if dim < len(input_shape):
            input_linear_index += (offsets % input_shape[dim]) * input_strides[dim]
        if dim < len(other_shape):
            other_linear_index += (offsets % other_shape[dim]) * other_strides[dim]
        offsets //= output_shape[dim]

    input_values = tl.load(input_ptr + input_linear_index, mask=offsets < output_shape[0], other=0.0)
    other_values = tl.load(other_ptr + other_linear_index, mask=offsets < output_shape[0], other=0.0)

    # Perform the multiplication
    output_values = input_values * other_values

    # Store the result in the output tensor
    tl.store(out_ptr + output_linear_index, output_values, mask=offsets < output_shape[0])

import torch
import triton
import triton.language as tl

def mul(input, other, *, out=None):
    # Determine the output shape and strides
    input_shape = input.shape
    other_shape = other.shape if isinstance(other, torch.Tensor) else (1,)
    output_shape = torch.broadcast_shapes(input_shape, other_shape)
    
    # Determine the data types and promote them if necessary
    input_dtype = input.dtype
    other_dtype = other.dtype if isinstance(other, torch.Tensor) else torch.get_default_dtype()
    output_dtype = torch.promote_types(input_dtype, other_dtype)
    
    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=output_dtype, device=input.device)
    
    # Determine the strides for the input, other, and output tensors
    input_strides = input.stride()
    other_strides = other.stride() if isinstance(other, torch.Tensor) else (0,)
    output_strides = out.stride()
    
    # Launch the Triton kernel
    grid = (triton.cdiv(out.numel(), 1024),)
    mul_kernel[grid](
        input.data_ptr(),  # Pointer to the input tensor
        other.data_ptr() if isinstance(other, torch.Tensor) else other,  # Pointer to the other tensor or number
        out.data_ptr(),  # Pointer to the output tensor
        input_shape,  # Shape of the input tensor
        other_shape,  # Shape of the other tensor
        output_shape,  # Shape of the output tensor
        input_strides,  # Strides of the input tensor
        other_strides,  # Strides of the other tensor
        output_strides,  # Strides of the output tensor
        input_dtype,  # Data type of the input tensor
        other_dtype,  # Data type of the other tensor
        output_dtype,  # Data type of the output tensor
        BLOCK_SIZE=1024,  # Block size for parallelization
    )
    
    return out
