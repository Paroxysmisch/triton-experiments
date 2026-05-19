import triton
import triton.language as tl
import torch
import math

# Kernel for element-wise multiplication of two tensors
@triton.jit
def mul_tensor_kernel(input_ptr, other_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of two tensors.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor (input).
    - out_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_tensor = tl.load(input_ptr + offsets, mask=mask)
    other_tensor = tl.load(other_ptr + offsets, mask=mask)
    out_tensor = input_tensor * other_tensor
    tl.store(out_ptr + offsets, out_tensor, mask=mask)

# Kernel for multiplication with a scalar
@triton.jit
def mul_scalar_kernel(input_ptr, scalar: tl.constexpr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of a tensor with a scalar.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - scalar: Scalar value to multiply with.
    - out_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_tensor = tl.load(input_ptr + offsets, mask=mask)
    out_tensor = input_tensor * scalar
    tl.store(out_ptr + offsets, out_tensor, mask=mask)

# Wrapper function for multiplication
def mul(input, other, *, out=None):
    """
    A wrapper function to invoke the appropriate Triton kernel for multiplication
    based on the type of input 'other' (tensor or scalar).
    
    Parameters:
    - input: The input tensor.
    - other: The tensor or number to multiply input by.
    - out: Optional output tensor.
    
    Returns:
    - Tensor: The resulting tensor after performing the multiplication.
    """
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    if isinstance(other, torch.Tensor):
        # Call kernel for tensor-tensor multiplication
        mul_tensor_kernel[(grid_size, 1, 1)](input, other, out, n_elements, block_size)
    else:
        # Call kernel for tensor-scalar multiplication
        mul_scalar_kernel[(grid_size, 1, 1)](input, other, out, n_elements, block_size)
    
    return out
