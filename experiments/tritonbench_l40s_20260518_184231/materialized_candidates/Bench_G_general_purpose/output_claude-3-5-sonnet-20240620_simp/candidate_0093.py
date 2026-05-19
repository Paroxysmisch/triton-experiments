import torch
import triton
import triton.language as tl

@triton.jit
def f8_to_f16_kernel(
    input_ptr,  # pointer to input int8 tensor
    output_ptr,  # pointer to output float16 tensor
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # size of parallel block processing
):
    # Calculate pid (program ID) and the block of elements to process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data with mask
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert int8 to float16 using bit manipulation
    # First, convert to int16 to have more bits for manipulation
    x = x.to(tl.int16)
    
    # Extract sign, exponent, and mantissa
    sign = (x >> 7) & 1
    exp = (x >> 4) & 0x7
    mantissa = x & 0xF
    
    # Adjust for float16 format
    # Move sign to bit 15
    sign = sign << 15
    # Adjust exponent bias and shift
    exp = ((exp + 15 - 7) << 10)
    # Shift mantissa to correct position
    mantissa = mantissa << 6
    
    # Combine components
    result = sign | exp | mantissa
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def f16_to_f8_kernel(
    input_ptr,  # pointer to input float16 tensor
    output_ptr,  # pointer to output int8 tensor
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # size of parallel block processing
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Extract components from float16
    sign = (x >> 15) & 1
    exp = (x >> 10) & 0x1F
    mantissa = (x >> 6) & 0xF  # Take only top 4 bits of mantissa
    
    # Adjust exponent bias for float8
    exp = tl.max(tl.min(exp - (15 - 7), 7), 0)
    
    # Combine components for float8
    result = (sign << 7) | (exp << 4) | mantissa
    
    # Convert to int8 and store
    tl.store(output_ptr + offsets, result.to(tl.int8), mask=mask)

# Python wrapper functions
def f8_to_f16(x: torch.Tensor) -> torch.Tensor:
    """
    Convert float8 (stored as int8) to float16
    
    Args:
        x: Input tensor in int8 format representing float8 values
        
    Returns:
        Converted tensor in float16 format
    """
    assert x.dtype == torch.int8, "Input tensor must be int8"
    output = torch.empty_like(x, dtype=torch.float16)
    n_elements = output.numel()
    
    # Configure grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    f8_to_f16_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def f16_to_f8(x: torch.Tensor) -> torch.Tensor:
    """
    Convert float16 to float8 (stored as int8)
    
    Args:
        x: Input tensor in float16 format
        
    Returns:
        Converted tensor in int8 format representing float8 values
    """
    assert x.dtype == torch.float16, "Input tensor must be float16"
    output = torch.empty_like(x, dtype=torch.int8)
    n_elements = output.numel()
    
    # Configure grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    f16_to_f8_kernel[grid](
        x, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
