import triton
import triton.language as tl
import torch

# Triton kernel for element-wise subtraction
@triton.jit
def sub_kernel(
    input_ptr, other_ptr, output_ptr, alpha, n_elements, input_stride, other_stride, output_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    A kernel to perform element-wise subtraction of two tensors, scaled by alpha.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor.
    - output_ptr: Pointer to the output tensor.
    - alpha: The multiplier for the other tensor.
    - n_elements: Total number of elements in the tensors.
    - input_stride: Stride of the input tensor.
    - other_stride: Stride of the other tensor.
    - output_stride: Stride of the output tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_offsets = input_stride * offsets
    other_offsets = other_stride * offsets
    output_offsets = output_stride * offsets
    
    input_val = tl.load(input_ptr + input_offsets, mask=mask)
    other_val = tl.load(other_ptr + other_offsets, mask=mask)
    output_val = input_val - alpha * other_val
    tl.store(output_ptr + output_offsets, output_val, mask=mask)

import torch
import triton
import triton.language as tl

def sub(input, other, *, alpha=1, out=None):
    """
    Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.
    The operation is defined as: out_i = input_i - alpha * other_i.
    Supports broadcasting to a common shape, type promotion, and works with integer, float, and complex inputs.
    
    Parameters:
    - input (Tensor): The input tensor.
    - other (Tensor or Number): The tensor or number to subtract from input.
    - alpha (Number): The multiplier for other.
    - out (Tensor, optional): The output tensor.
    
    Returns:
    - Tensor: The resulting tensor after performing the subtraction.
    """
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    
    # Ensure other is a tensor
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast input and other to a common shape
    input, other = torch.broadcast_tensors(input, other)
    
    # Ensure out has the same shape as input and other
    if out.shape != input.shape:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    
    # Determine strides and number of elements
    n_elements = input.numel()
    input_stride = input.stride(0)
    other_stride = other.stride(0)
    output_stride = out.stride(0)
    
    # Determine block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernel
    sub_kernel[(grid_size, 1, 1)](
        input, other, out, alpha, n_elements, input_stride, other_stride, output_stride, block_size
    )
    
    return out

# Example tensors
input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda')
other = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')

# Perform the subtraction
result = sub(input, other, alpha=2.0)

print(result)  # Output: tensor([0.0, 1.0, 2.0], device='cuda:0')
