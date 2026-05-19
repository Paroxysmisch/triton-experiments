import torch
import triton
import triton.language as tl
from triton.language.math import erf, pow, tanh
import math

@triton.jit
def sub_gelu_none_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements, alpha,
    input_stride, other_stride, output_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and other
    input_offset = input_ptr + offsets * input_stride
    other_offset = other_ptr + offsets * other_stride
    x = tl.load(input_offset, mask=mask)
    y = tl.load(other_offset, mask=mask)

    # Compute subtraction with alpha scaling
    z = x - alpha * y
    
    # Convert to float32 for better precision
    z_f32 = z.to(tl.float32)
    
    # Compute GELU using error function (exact method)
    result = 0.5 * z_f32 * (1.0 + erf(z_f32 * 0.7071067811865476))  # 1/sqrt(2)
    
    # Store result
    output_offset = output_ptr + offsets * output_stride
    tl.store(output_offset, result, mask=mask)

@triton.jit
def sub_gelu_tanh_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements, alpha,
    input_stride, other_stride, output_stride,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and other
    input_offset = input_ptr + offsets * input_stride
    other_offset = other_ptr + offsets * other_stride
    x = tl.load(input_offset, mask=mask)
    y = tl.load(other_offset, mask=mask)

    # Compute subtraction with alpha scaling
    z = x - alpha * y
    
    # Convert to float32 for better precision
    z_f32 = z.to(tl.float32)
    
    # Constants for GELU approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coef = 0.044715
    
    # Compute GELU using tanh approximation
    inner = sqrt_2_over_pi * (z_f32 + coef * pow(z_f32, 3))
    result = 0.5 * z_f32 * (1.0 + tanh(inner))
    
    # Store result
    output_offset = output_ptr + offsets * output_stride
    tl.store(output_offset, result, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None):
    """
    Subtracts 'other' scaled by 'alpha' from 'input', then applies GELU activation.
    
    Args:
        input (Tensor): The input tensor
        other (Tensor or Number): The tensor or number to subtract from input
        alpha (Number, optional): The multiplier for other. Default: 1
        approximate (str, optional): The approximation method for GELU ('none' or 'tanh'). Default: 'none'
        out (Tensor, optional): The output tensor. Default: None
    
    Returns:
        Tensor: The result of GELU(input - alpha * other)
    """
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.full_like(input, other)
    
    # Input validation
    if input.device != other.device:
        raise ValueError("Input and other must be on the same device")
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be either 'none' or 'tanh'")
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise ValueError("Output tensor must have the same shape as input")
    
    # Get tensor dimensions and properties
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Get strides
    input_stride = input.stride(0) if input.stride(0) > 0 else 1
    other_stride = other.stride(0) if other.stride(0) > 0 else 1
    output_stride = out.stride(0) if out.stride(0) > 0 else 1

    # Choose kernel based on approximation method
    kernel = sub_gelu_none_kernel if approximate == 'none' else sub_gelu_tanh_kernel
    
    # Launch kernel
    kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        n_elements,
        alpha,
        input_stride,
        other_stride,
        output_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
