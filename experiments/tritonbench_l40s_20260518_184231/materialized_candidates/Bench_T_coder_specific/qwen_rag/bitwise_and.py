import triton
import triton.language as tl
import torch
import math

# Kernel for bitwise AND operation on two tensors (A and B) element-wise and store the result in tensor C.
@triton.jit
def bitwise_and_func_tensor(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform bitwise AND operation on two tensors A and B, and store the result in C.
    
    Parameters:
    - A_ptr: Pointer to the tensor A (input).
    - B_ptr: Pointer to the tensor B (input).
    - C_ptr: Pointer to the tensor C (output).
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# Kernel for bitwise AND operation on a tensor A and a constant scalar B
@triton.jit
def bitwise_and_func_scalar(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform bitwise AND operation on a tensor A and a scalar B, and store the result in C.
    
    Parameters:
    - A_ptr: Pointer to the tensor A (input).
    - B: Scalar value to perform AND operation with.
    - C_ptr: Pointer to the tensor C (output).
    - n_elements: Total number of elements in the tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# Wrapper function to choose between tensor-based and scalar-based bitwise AND
def bitwise_and(A, B, out=None):
    """
    A wrapper function to invoke the appropriate Triton kernel for bitwise AND operation
    based on the type of input B (tensor or scalar).
    
    Parameters:
    - A: The first input tensor.
    - B: The second input (either a tensor or scalar).
    - out (Tensor, optional): The output tensor. If None, a new tensor will be created.
    
    Returns:
    - C: The resulting tensor after performing the bitwise AND.
    """
    if out is None:
        C = torch.empty_like(A)
    else:
        C = out
    
    n_elements = C.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    if isinstance(B, torch.Tensor):
        # Call kernel for tensor-tensor bitwise AND
        bitwise_and_func_tensor[(grid_size, 1, 1)](A, B, C, n_elements, block_size)
    else:
        # Call kernel for tensor-scalar bitwise AND
        bitwise_and_func_scalar[(grid_size, 1, 1)](A, B, C, n_elements, block_size)   
    
    return C
