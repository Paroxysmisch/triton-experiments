import triton
import triton.language as tl
import torch

# Constants for FP4 format
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x1
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3f000000
F4_E2M1_EXP_BIAS = 1
F32_EXP_BIAS = 127

@triton.jit
def _fp4_packed_to_bf16(packed_fp4):
    """Helper function to convert a packed FP4 value to BF16."""
    # Extract components
    sign = (packed_fp4 & SIGN_MASK_F4) >> 3
    exp = (packed_fp4 & 0x6) >> 1
    mantissa = packed_fp4 & MANTISSA_MASK_F4
    
    # Handle special cases
    is_zero = exp == 0 and mantissa == 0
    is_denormal = exp == 0 and mantissa == 1
    
    # Convert to BF16 format
    if is_zero:
        return ZERO_BITS_F32
    if is_denormal:
        return ZERO_POINT_FIVE_BITS_F32 | (sign << 31)
    
    # Normal number conversion
    adjusted_exp = exp + F32_EXP_BIAS - F4_E2M1_EXP_BIAS
    result = (sign << 31) | (adjusted_exp << 23) | (mantissa << 22)
    return result

@triton.jit
def triton_f4_to_bf16_kernel(
    x_ptr,  # pointer to input packed FP4 values
    output_ptr,  # pointer to output BF16 values
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Compute linear index for this thread
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Load offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load packed FP4 values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Convert each value to BF16
    output = _fp4_packed_to_bf16(x)
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_f4_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """
    Convert packed FP4 tensor to BF16 format using Triton.
    
    Args:
        x: Input tensor in packed FP4 format
    Returns:
        Tensor in BF16 format
    """
    # Ensure input is on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Calculate dimensions
    n_elements = x.numel()
    output = torch.empty(n_elements, dtype=torch.bfloat16, device=x.device)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    triton_f4_to_bf16_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
