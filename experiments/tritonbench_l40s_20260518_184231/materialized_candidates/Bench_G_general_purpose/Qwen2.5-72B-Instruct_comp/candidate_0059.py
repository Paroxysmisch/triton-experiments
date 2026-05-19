import triton
import triton.language as tl

# Constants for FP4 and FP32
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x7
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
EBITS_F4_E2M1 = 2
MBITS_F4_E2M1 = 3
EBITS_F32 = 8
MBITS_F32 = 23
BIAS_F4 = 7
BIAS_F32 = 127

@triton.jit
def _fp4_packed_to_bf16(packed_fp4: tl.int8) -> tl.bfloat16:
    # Extract low and high FP4 values
    low_fp4 = packed_fp4 & 0xF
    high_fp4 = (packed_fp4 >> 4) & 0xF

    # Function to convert a single FP4 to BF16
    def fp4_to_bf16(fp4: tl.int8) -> tl.bfloat16:
        sign = (fp4 & SIGN_MASK_F4) >> 3
        mantissa = fp4 & MANTISSA_MASK_F4
        exponent = (fp4 >> MBITS_F4_E2M1) & ((1 << EBITS_F4_E2M1) - 1)

        if exponent == 0:
            if mantissa == 0:
                return tl.bfloat16(sign << 7)  # Zero
            else:
                return tl.bfloat16((sign << 7) | (1 << 6))  # Denormal (0.5)
        else:
            exponent = (exponent + BIAS_F32 - BIAS_F4) << MBITS_F32
            mantissa = mantissa << (MBITS_F32 - MBITS_F4_E2M1)
            return tl.bfloat16((sign << 7) | exponent | mantissa)

    low_bf16 = fp4_to_bf16(low_fp4)
    high_bf16 = fp4_to_bf16(high_fp4)

    return low_bf16, high_bf16

@triton.jit
def triton_f4_to_bf16_kernel(
    packed_fp4_ptr: tl.pointer('i8'),
    bf16_ptr: tl.pointer('bf16'),
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < size
    packed_fp4 = tl.load(packed_fp4_ptr + offsets, mask=mask, other=0)

    # Convert packed FP4 to BF16
    low_bf16, high_bf16 = _fp4_packed_to_bf16(packed_fp4)

    # Store the results
    tl.store(bf16_ptr + (offsets * 2), low_bf16, mask=mask)
    tl.store(bf16_ptr + (offsets * 2 + 1), high_bf16, mask=mask)

def triton_f4_to_bf16(packed_fp4_tensor):
    assert packed_fp4_tensor.is_cuda, "Input tensor must be on a CUDA device"
    assert packed_fp4_tensor.is_contiguous(), "Input tensor must be contiguous"

    size = packed_fp4_tensor.numel()
    bf16_tensor = torch.empty(size * 2, dtype=torch.bfloat16, device=packed_fp4_tensor.device)

    grid = (triton.cdiv(size, 1024),)
    BLOCK_SIZE = 1024

    triton_f4_to_bf16_kernel[grid](
        packed_fp4_tensor, bf16_tensor, size, BLOCK_SIZE
    )

    return bf16_tensor
