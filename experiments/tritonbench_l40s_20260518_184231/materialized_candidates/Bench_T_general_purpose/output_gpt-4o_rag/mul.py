import triton
import triton.language as tl
import torch
import math

# Kernel for element-wise multiplication of two tensors
@triton.jit
def multiply_tensors_kernel(input_ptr, other_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of two tensors, input and other, and store the result in out.

    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor.
    - out_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the output tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    out_val = input_val * other_val
    tl.store(out_ptr + offsets, out_val, mask=mask)

# Kernel for element-wise multiplication of a tensor and a scalar
@triton.jit
def multiply_tensor_scalar_kernel(input_ptr, scalar: tl.constexpr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of a tensor and a scalar, and store the result in out.

    Parameters:
    - input_ptr: Pointer to the input tensor.
    - scalar: Scalar value to multiply with the tensor.
    - out_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the output tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    out_val = input_val * scalar
    tl.store(out_ptr + offsets, out_val, mask=mask)

# Wrapper function for multiplication
def mul(input, other, *, out=None):
    """
    A wrapper function to perform multiplication of an input tensor with another tensor or scalar.

    Parameters:
    - input: The input tensor.
    - other: The tensor or scalar to multiply with.
    - out: Optional output tensor.

    Returns:
    - out: The resulting tensor after multiplication.
    """
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = out.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    if isinstance(other, torch.Tensor):
        # Ensure the shapes are broadcastable
        input, other = torch.broadcast_tensors(input, other)
        multiply_tensors_kernel[(grid_size,)](input, other, out, n_elements, block_size)
    else:
        # Scalar multiplication
        multiply_tensor_scalar_kernel[(grid_size,)](input, other, out, n_elements, block_size)
    
    return out
