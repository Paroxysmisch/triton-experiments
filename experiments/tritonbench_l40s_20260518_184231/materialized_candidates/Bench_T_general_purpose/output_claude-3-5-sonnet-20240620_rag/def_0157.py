import triton
import triton.language as tl
import torch
from typing import Tuple

@triton.jit
def signbit_bitwise_and_kernel(
    input_ptr, 
    other_ptr,
    signbit_ptr,
    bitwise_result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID and compute the block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    
    # Compute sign bit check (True for negative values including -0)
    # We can use bitwise operations to check the sign bit
    signbit_result = tl.where(input_vals < 0, True, False)
    
    # Compute bitwise AND
    bitwise_result = input_vals & other_vals
    
    # Store results
    tl.store(signbit_ptr + offsets, signbit_result, mask=mask)
    tl.store(bitwise_result_ptr + offsets, bitwise_result, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and bitwise AND operation on input tensors.
    
    Args:
        input (Tensor): The input tensor
        other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types
        
    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Tensor with True for negative values (including -0)
            - Tensor with the result of bitwise AND operation
    """
    # Ensure inputs have the same shape
    assert input.shape == other.shape, "Input tensors must have the same shape"
    assert other.dtype in [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64], \
           "Other tensor must be of integral or boolean type"
    
    # Create output tensors
    n_elements = input.numel()
    signbit_result = torch.empty(input.shape, dtype=torch.bool, device=input.device)
    bitwise_result = torch.empty_like(other)
    
    # Calculate launch grid
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    signbit_bitwise_and_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        signbit_result.data_ptr(),
        bitwise_result.data_ptr(),
        n_elements,
        BLOCK_SIZE
    )
    
    return signbit_result, bitwise_result
