import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def signbit_bitwise_and_kernel(
    input_ptr,
    other_ptr,
    signbit_output_ptr,
    bitwise_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    # Compute block start/end
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    
    # Compute signbit (check if the sign bit is set)
    signbit_result = x < 0
    
    # Compute bitwise AND
    bitwise_result = tl.bitwise_and(tl.cast(x, tl.int32), tl.cast(y, tl.int32))
    
    # Store results
    tl.store(signbit_output_ptr + offsets, signbit_result, mask=mask)
    tl.store(bitwise_output_ptr + offsets, bitwise_result, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and bitwise AND operation on input tensors.
    
    Args:
        input (Tensor): The input tensor
        other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types
        
    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Tensor with True where input is negative (including -0)
            - Tensor with bitwise AND result of input and other
    """
    # Input validation
    if not input.is_contiguous():
        input = input.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()
    
    assert input.shape == other.shape, "Input tensors must have the same shape"
    assert input.device.type == "cuda", "Input tensor must be on GPU"
    assert other.device.type == "cuda", "Other tensor must be on GPU"
    
    # Prepare output tensors
    n_elements = input.numel()
    signbit_output = torch.empty_like(input, dtype=torch.bool)
    bitwise_output = torch.empty_like(other, dtype=other.dtype)
    
    # Configure grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    signbit_bitwise_and_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        signbit_output_ptr=signbit_output,
        bitwise_output_ptr=bitwise_output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return signbit_output, bitwise_output
