import triton
import triton.language as tl
import torch
import math

# Kernel for bitwise AND operation on tensors
@triton.jit
def bitwise_and_func_tensor(input_ptr, other_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform bitwise AND operation on two tensors input and other, and store the result in out.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor.
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
    out_tensor = input_tensor & other_tensor
    tl.store(out_ptr + offsets, out_tensor, mask=mask)

# Wrapper function for bitwise AND
def bitwise_and(input, other, *, out=None):
    """
    A wrapper function to perform bitwise AND operation on input and other tensors.
    
    Parameters:
    - input: The first input tensor (must be integral or Boolean).
    - other: The second input tensor (must be integral or Boolean).
    - out (Tensor, optional): The output tensor.
    
    Returns:
    - Tensor: The resulting tensor after performing the bitwise AND.
    """
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Call kernel for tensor-tensor bitwise AND
    bitwise_and_func_tensor[(grid_size, 1, 1)](input, other, out, n_elements, block_size)
    
    return out
