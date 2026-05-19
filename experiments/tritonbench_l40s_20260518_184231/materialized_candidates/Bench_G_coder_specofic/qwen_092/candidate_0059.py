import triton
import torch

@triton.jit
def _fp4_packed_to_bf16(x: float32, y: float32) -> float32:
    # Masks for extracting parts of FP4 numbers
    SIGN_MASK_F4 = 0x8000
    MANTISSA_MASK_F4 = 0x7FFF
    # Bit patterns for special FP32 values (zero and 0.5)
    ZERO_BITS_F32 = 0x00000000
    ZERO_POINT_FIVE_BITS_F32 = 0x3F800000
    # Exponent and mantissa specifications for FP4 and FP32
    EBITS_F4_E2M1 = 3
    MBITS_F4_E2M1 = 10
    EBITS_F32 = 8
    MBITS_F32 = 23
    # Bias constants to correct exponent value differences between formats
    FP4_BIAS = 14
    FP32_BIAS = 127

    # Extract sign, exponent, and mantissa from FP4
    sign = (x & SIGN_MASK_F4) >> (EBITS_F4_E2M1 + MBITS_F4_E2M1)
    exponent = (x & ((1 << (EBITS_F4_E2M1 + MBITS_F4_E2M1)) - 1)) >> MBITS_F4_E2M1
    mantissa = x & MANTISSA_MASK_F4

    # Handle special cases
    if exponent == 0:
        if mantissa == 0:
            # Zero
            return float32.from_bits(ZERO_BITS_F32)
        elif mantissa == 0x4000:
            # Denormal (0.5)
            return float32.from_bits(ZERO_POINT_FIVE_BITS_F32)
        else:
            # Subnormal
            exponent = 1
            mantissa = (mantissa << 1) | 1
    else:
        # Normalized
        exponent = exponent - FP4_BIAS + FP32_BIAS

    # Reconstruct value in FP32 format
    reconstructed_value = (sign << (EBITS_F32 + MBITS_F32)) | (exponent << MBITS_F32) | mantissa

    # Convert to BF16
    return float32.from_bits(reconstructed_value).astype(float16)

@triton.jit
def triton_f4_to_bf16_kernel(x_ptr: float32, y_ptr: float16, n_elements: int32, BLOCK_SIZE: int32 = 256):
    pid = triton.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)

    for i in range(block_start, block_end):
        x = x_ptr[i]
        y_ptr[i] = _fp4_packed_to_bf16(x, 0.0)

def triton_f4_to_bf16(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.dtype == torch.float32, "Input tensor must be of type float32"

    y = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')
    grid = (triton.cdiv(x.numel(), 256),)
    triton_f4_to_bf16_kernel[grid](x, y, x.numel())
    return y
