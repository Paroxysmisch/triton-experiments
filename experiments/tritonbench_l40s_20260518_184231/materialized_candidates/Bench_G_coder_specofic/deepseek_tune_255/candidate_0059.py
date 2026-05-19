import torch
import triton
import triton.language as tl

@triton.jit
def _fp4_packed_to_bf16(
    packed_low,
    packed_high,
    bf16_output_ptr,
    # constants
    SIGN_MASK_F4: tl.constexpr,
    MANTISSA_MASK_F4: tl.constexpr,
    ZERO_BITS_F32: tl.constexpr,
    ZERO_POINT_FIVE_BITS_F32: tl.constexpr,
    EBITS_F4_E2M1: tl.constexpr,
    MBITS_F4_E2M1: tl.constexpr,
    EBITS_F32: tl.constexpr,
    MBITS_F32: tl.constexpr,
    E2M1_BIAS: tl.constexpr,
):
    """Convert FP4 to BF16"""
    # extract sign, exponent, mantissa
    fp4_low = packed_low.to(tl.int32)
    fp4_high = packed_high.to(tl.int32)
    sign_low = fp4_low & SIGN_MASK_F4
    sign_high = fp4_high & SIGN_MASK_F4
    exponent_low = (fp4_low >> MBITS_F4_E2M1) & EBITS_F4_E2M1
    exponent_high = (fp4_high >> MBITS_F4_E2M1) & EBITS_F4_E2M1
    mantissa_low = fp4_low & MANTISSA_MASK_F4
    mantissa_high = fp4_high & MANTISSA_MASK_F4
    # handle special cases
    zero_low = exponent_low == 0 and mantissa_low == 0
    zero_high = exponent_high == 0 and mantissa_high == 0
    denormal_low = exponent_low == 0 and mantissa_low != 0
    denormal_high = exponent_high == 0 and mantissa_high != 0
    inf_low = exponent_low == EBITS_F4_E2M1 and mantissa_low == 0
    inf_high = exponent_high == EBITS_F4_E2M1 and mantissa_high == 0
    # convert from FP4 to FP32
    exponent_low -= E2M1_BIAS
    exponent_high -= E2M1_BIAS
    fp32_low = sign_low | (exponent_low << (MBITS_F32 - 1)) | (mantissa_low << (MBITS_F32 - MBITS_F4_E2M1))
    fp32_high = sign_high | (exponent_high << (MBITS_F32 - 1)) | (mantissa_high << (MBITS_F32 - MBITS_F4_E2M1))
    fp32_low = tl.where(zero_low, ZERO_BITS_F32, fp32_low)
    fp32_low = tl.where(denormal_low, ZERO_POINT_FIVE_BITS_F32, fp32_low)
    fp32_high = tl.where(zero_high, ZERO_BITS_F32, fp32_high)
    fp32_high = tl.where(denormal_high, ZERO_POINT_FIVE_BITS_F32, fp32_high)
    fp32_low = tl.where(inf_low, 0x7f800000, fp32_low)
    fp32_high = tl.where(inf_high, 0x7f800000, fp32_high)
    # convert from FP32 to BF16
    bf16_low = tl.libdevice.bitcast(fp32_low, tl.bfloat16)
    bf16_high = tl.libdevice.bitcast(fp32_high, tl.bfloat16)
    # store
    tl.store(bf16_output_ptr, bf16_low)
    tl.store(bf16_output_ptr + 1, bf16_high)

@triton.jit
def triton_f4_to_bf16_kernel(
    packed_f4_ptr,
    bf16_output_ptr,
    # parameters
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    # constants
    SIGN_MASK_F4: tl.constexpr,
    MANTISSA_MASK_F4: tl.constexpr,
    ZERO_BITS_F32: tl.constexpr,
    ZERO_POINT_FIVE_BITS_F32: tl.constexpr,
    EBITS_F4_E2M1: tl.constexpr,
    MBITS_F4_E2M1: tl.constexpr,
    EBITS_F32: tl.constexpr,
    MBITS_F32: tl.constexpr,
    E2M1_BIAS: tl.constexpr,
):
    """Convert FP4 to BF16"""
    # program id
    pid = tl.program_id(0)
    # loop over elements
    for i in range(pid, num_elements, BLOCK_SIZE):
        _fp4_packed_to_bf16(
            tl.load(packed_f4_ptr + i),
            tl.load(packed_f4_ptr + i + 1),
            bf16_output_ptr + i,
            # constants
            SIGN_MASK_F4,
            MANTISSA_MASK_F4,
            ZERO_BITS_F32,
            ZERO_POINT_FIVE_BITS_F32,
            EBITS_F4_E2M1,
            MBITS_F4_E2M1,
            EBITS_F32,
            MBITS_F32,
            E2M1_BIAS,
        )

def triton_f4_to_bf16(packed_f4):
    """Convert packed FP4 to BF16"""
    assert packed_f4.is_cuda and packed_f4.ndim == 1
    num_elements = packed_f4.numel()
    bf16 = torch.empty((num_elements // 2,), dtype=torch.bfloat16, device="cuda")
    # grid
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    # constants
    SIGN_MASK_F4 = 0x8000
    MANTISSA_MASK_F4 = 0x7FFF
    ZERO_BITS_F32 = 0x0000
    ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
    EBITS_F4_E2M1 = 0x2
    MBITS_F4_E2M1 = 0xA
    EBITS_F32 = 0xFF
    MBITS_F32 = 0x1F
    E2M1_BIAS = 1
    # kernel
    triton_f4_to_bf16_kernel[grid](
        packed_f4,
        bf16,
        num_elements,
        # constants
        SIGN_MASK_F4,
        MANTISSA_MASK_F4,
        ZERO_BITS_F32,
        ZERO_POINT_FIVE_BITS_F32,
        EBITS_F4_E2M1,
        MBITS_F4_E2M1,
        EBITS_F32,
