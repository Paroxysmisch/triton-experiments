import triton
import triton.language as tl
import torch

# Constants for FP4
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x3
EXPONENT_MASK_F4 = 0x4
EXPONENT_BIAS_F4 = 3

# Constants for FP32 and BF16
EXPONENT_BIAS_F32 = 127
EXPONENT_BIAS_BF16 = 127

ZERO_BITS_F32 = 0x00000000
ZERO_POINT_FIVE_BITS_F32 = 0x3f000000

# Helper function to convert FP4 packed values to BF16
@triton.jit
def _fp4_packed_to_bf16(packed_values, bf16_output, block_start):
    # Load packed FP4 values
    packed = tl.load(packed_values + block_start)

    # Separate low and high 4-bit values
    low_fp4 = packed & 0xF
    high_fp4 = (packed >> 4) & 0xF

    # Process each FP4 value separately
    for i, fp4_value in enumerate([low_fp4, high_fp4]):
        sign = (fp4_value & SIGN_MASK_F4) << 28
        exponent = (fp4_value & EXPONENT_MASK_F4) >> 2
        mantissa = fp4_value & MANTISSA_MASK_F4

        # Handle zero and denormal (0.5) cases
        if exponent == 0 and mantissa == 0:
            fp32_bits = ZERO_BITS_F32
        elif exponent == 0 and mantissa != 0:
            fp32_bits = ZERO_POINT_FIVE_BITS_F32
        else:
            # Adjust exponent from FP4 to FP32/BF16
            adjusted_exponent = (exponent - EXPONENT_BIAS_F4 + EXPONENT_BIAS_BF16) << 23
            fp32_bits = sign | adjusted_exponent | (mantissa << 21)

        # Convert FP32 to BF16 by truncating mantissa
        bf16_value = fp32_bits >> 16
        tl.store(bf16_output + block_start + i, bf16_value)

# Main Triton kernel for processing
@triton.jit
def triton_f4_to_bf16_kernel(packed_values_ptr, bf16_output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate block index
    block_start = tl.program_id(0) * BLOCK_SIZE

    # Guard against out-of-bounds
    if block_start < num_elements:
        _fp4_packed_to_bf16(packed_values_ptr, bf16_output_ptr, block_start)

# Python wrapper for the kernel
def triton_f4_to_bf16(packed_values):
    assert packed_values.is_cuda and packed_values.is_contiguous(), "Input must be CUDA and contiguous"
    
    num_elements = packed_values.numel() * 2  # Each packed byte contains 2 FP4 values
    bf16_output = torch.empty(num_elements, dtype=torch.bfloat16, device=packed_values.device)

    # Define grid size
    BLOCK_SIZE = 128
    grid = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    triton_f4_to_bf16_kernel[grid](packed_values, bf16_output, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    return bf16_output
