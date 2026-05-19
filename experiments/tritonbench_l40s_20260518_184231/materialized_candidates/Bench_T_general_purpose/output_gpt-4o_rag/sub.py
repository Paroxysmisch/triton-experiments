import triton
import triton.language as tl
import torch
import math

# Kernel for the subtraction operation
@triton.jit
def subtract_kernel(input_ptr, other_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform the operation: out_i = input_i - alpha * other_i
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor.
    - output_ptr: Pointer to the output tensor.
    - alpha: Scalar multiplier for the other tensor.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    result = input_val - alpha * other_val
    tl.store(output_ptr + offsets, result, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    """
    A wrapper function to perform element-wise subtraction of a scaled `other` from `input`.
    
    Parameters:
    - input: The input tensor.
    - other: The tensor or number to subtract from input.
    - alpha: The multiplier for other.
    - out: The output tensor.
    
    Returns:
    - The resulting tensor after performing the operation.
    """
    # Handle broadcasting and type promotion
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    input, other = torch.broadcast_tensors(input, other)
    
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = out.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernel
    subtract_kernel[(grid_size,)](input, other, out, alpha, n_elements, block_size)
    
    return out
