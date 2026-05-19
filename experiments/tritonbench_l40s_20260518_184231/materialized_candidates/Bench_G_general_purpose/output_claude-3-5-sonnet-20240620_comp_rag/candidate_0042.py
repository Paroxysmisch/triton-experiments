import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    # Pointers to input/output tensors
    input_ptr,
    output_ptr,
    # Scalar exponent value
    exponent,
    # Shape and stride information
    n_elements,
    input_stride,
    output_stride,
    # Block size for tiling
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Calculate input/output addresses
    input_block_ptr = input_ptr + offsets * input_stride
    output_block_ptr = output_ptr + offsets * output_stride
    
    # Load input data
    x = tl.load(input_block_ptr, mask=mask)
    
    # Compute power using Triton's pow function
    result = tl.pow(x, exponent)
    
    # Store the result
    tl.store(output_block_ptr, result, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    input_tensor: torch.Tensor,
    exponent: float,
    output_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Input validation
    assert input_tensor.dim() == 1, "Input tensor must be 1-dimensional"
    
    # Create output tensor if not provided
    if output_tensor is None:
        output_tensor = torch.empty_like(input_tensor)
    
    # Get tensor properties
    n_elements = input_tensor.numel()
    
    # Calculate optimal block size (power of 2)
    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        input_ptr=input_tensor.data_ptr(),
        output_ptr=output_tensor.data_ptr(),
        exponent=exponent,
        n_elements=n_elements,
        input_stride=input_tensor.stride(0),
        output_stride=output_tensor.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor
