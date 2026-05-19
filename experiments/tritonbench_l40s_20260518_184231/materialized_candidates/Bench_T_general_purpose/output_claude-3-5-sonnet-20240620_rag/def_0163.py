import torch
import triton
import triton.language as tl
import math
from typing import Tuple

@triton.jit
def cos_signbit_kernel(
    input_ptr,
    cos_output_ptr,
    sign_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate offset for current block
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offset < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offset, mask=mask)
    
    # Compute cosine
    cos_result = tl.cos(x.to(tl.float32))
    
    # Compute sign bit (1 for negative, 0 for positive)
    sign_bit = cos_result < 0
    
    # Store results
    tl.store(cos_output_ptr + offset, cos_result, mask=mask)
    tl.store(sign_output_ptr + offset, sign_bit, mask=mask)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cosine of each element in the input tensor and determines the sign bit.
    
    Args:
        input (Tensor): The input tensor for which the cosine and sign bit are computed.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The cosine of each element
            - A boolean tensor indicating the sign bit of each cosine result (True for negative)
    """
    # Input validation
    if not input.is_contiguous():
        input = input.contiguous()
    
    # Prepare output tensors
    cos_output = torch.empty_like(input)
    sign_output = torch.empty_like(input, dtype=torch.bool)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate optimal block size
    block_size = triton.next_power_of_2(min(max(n_elements, 128), 4096))
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch kernel
    cos_signbit_kernel[(grid_size,)](
        input,
        cos_output,
        sign_output,
        n_elements,
        block_size
    )
    
    return cos_output, sign_output
