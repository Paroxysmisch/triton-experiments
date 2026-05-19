import triton
import triton.language as tl
import torch
import math

# Kernel for tensor-tensor multiplication and subtraction
@triton.jit
def mul_sub_tensor_tensor(input_ptr, other_mul_ptr, other_sub_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication and subtraction of tensors.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_mul_ptr: Pointer to the tensor to multiply with `input`.
    - other_sub_ptr: Pointer to the tensor to subtract from the multiplication result.
    - out_ptr: Pointer to the output tensor.
    - alpha: The multiplier for `other_sub`.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other_mul = tl.load(other_mul_ptr + offsets, mask=mask)
    other_sub = tl.load(other_sub_ptr + offsets, mask=mask)
    result = (input * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel for tensor-scalar multiplication and tensor subtraction
@triton.jit
def mul_sub_tensor_scalar(input_ptr, other_mul: tl.constexpr, other_sub_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication with a scalar and subtraction of a tensor.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_mul: Scalar value to multiply with `input`.
    - other_sub_ptr: Pointer to the tensor to subtract from the multiplication result.
    - out_ptr: Pointer to the output tensor.
    - alpha: The multiplier for `other_sub`.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other_sub = tl.load(other_sub_ptr + offsets, mask=mask)
    result = (input * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel for tensor multiplication and scalar subtraction
@triton.jit
def mul_sub_tensor_tensor_scalar(input_ptr, other_mul_ptr, other_sub: tl.constexpr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication of tensors and subtraction of a scalar.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_mul_ptr: Pointer to the tensor to multiply with `input`.
    - other_sub: Scalar value to subtract from the multiplication result.
    - out_ptr: Pointer to the output tensor.
    - alpha: The multiplier for `other_sub`.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other_mul = tl.load(other_mul_ptr + offsets, mask=mask)
    result = (input * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel for tensor-scalar multiplication and scalar subtraction
@triton.jit
def mul_sub_tensor_scalar_scalar(input_ptr, other_mul: tl.constexpr, other_sub: tl.constexpr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform element-wise multiplication with a scalar and subtraction of a scalar.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_mul: Scalar value to multiply with `input`.
    - other_sub: Scalar value to subtract from the multiplication result.
    - out_ptr: Pointer to the output tensor.
    - alpha: The multiplier for `other_sub`.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    result = (input * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, result, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    """
    Multiplies the input tensor by another tensor or number, then subtracts another tensor or number from the result,
    scaled by a given alpha. This operation is performed element-wise.
    
    Parameters:
    - input (Tensor): The input tensor to be multiplied.
    - other_mul (Tensor or Number): The tensor or number to multiply with `input`.
    - other_sub (Tensor or Number): The tensor or number to subtract from the multiplication result.
    - alpha (Number, optional): The multiplier for `other_sub`. Default is 1.
    - out (Tensor, optional): The output tensor.
    
    Returns:
    - Tensor: The resulting tensor after performing the operation.
    """
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = out.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    if isinstance(other_mul, torch.Tensor) and isinstance(other_sub, torch.Tensor):
        mul_sub_tensor_tensor[(grid_size, 1, 1)](input, other_mul, other_sub, out, alpha, n_elements, block_size)
    elif isinstance(other_mul, torch.Tensor) and isinstance(other_sub, (int, float)):
        mul_sub_tensor_tensor_scalar[(grid_size, 1, 1)](input, other_mul, other_sub, out, alpha, n_elements, block_size)
    elif isinstance(other_mul, (int, float)) and isinstance(other_sub, torch.Tensor):
        mul_sub_tensor_scalar[(grid_size, 1, 1)](input, other_mul, other_sub, out, alpha, n_elements, block_size)
    elif isinstance(other_mul, (int, float)) and isinstance(other_sub, (int, float)):
        mul_sub_tensor_scalar_scalar[(grid_size, 1, 1)](input, other_mul, other_sub, out, alpha, n_elements, block_size)
    else:
        raise TypeError("Unsupported input types for `other_mul` and `other_sub`.")
    
    return out

# Example usage
input = torch.tensor([1.0, 2.0, 3.0])
other_mul = torch.tensor([2.0, 2.0, 2.0])
other_sub = torch.tensor([1.0, 1.0, 1.0])
alpha = 2

# Tensor-tensor case
result = mul_sub(input, other_mul, other_sub, alpha)
print(result)  # Expected: [1.0, 3.0, 5.0]

# Tensor-scalar case
other_sub = 1.0
result = mul_sub(input, other_mul, other_sub, alpha)
print(result)  # Expected: [1.0, 3.0, 5.0]

# Scalar-tensor case
other_mul = 2.0
other_sub = torch.tensor([1.0, 1.0, 1.0])
result = mul_sub(input, other_mul, other_sub, alpha)
print(result)  # Expected: [1.0, 3.0, 5.0]

# Scalar-scalar case
other_mul = 2.0
other_sub = 1.0
result = mul_sub(input, other_mul, other_sub, alpha)
print(result)  # Expected: [1.0, 3.0, 5.0]
