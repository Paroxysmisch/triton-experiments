import triton
import triton.language as tl

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,  # *uint8
    s_ptr,  # *float32
    output_ptr,  # *bfloat16
    n_elements_in,  # size_t
    BLOCK_SIZE_IN: tl.constexpr,
    SIGN_MASK_F4: tl.constexpr,
    EXP_MASK_F4: tl.constexpr,
    MANTISSA_MASK_F4: tl.constexpr,
    EXP_BIAS_F4: tl.constexpr,
    EXP_BIAS_BF16: tl.constexpr,
    SIGN_MASK_BF16: tl.constexpr,
    EXP_MASK_BF16: tl.constexpr,
    MANTISSA_MASK_BF16: tl.constexpr,
    ZERO_BITS_F32: tl.constexpr,
    DENORMAL_THRESHOLD_F32: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_IN
    offsets = block_start + tl.arange(0, BLOCK_SIZE_IN)
    mask = offsets < n_elements_in

    # Load packed 4-bit values
    packed_x = tl.load(x_ptr + offsets // 2, mask=mask, other=0)
    x_lo = packed_x & 0x0F
    x_hi = (packed_x >> 4) & 0x0F

    # Load scale
    scale = tl.load(s_ptr)

    # Convert to fp32
    sign_f4 = (x_lo & SIGN_MASK_F4) << 27
    exp_f4 = (x_lo & EXP_MASK_F4) << 23
    mantissa_f4 = (x_lo & MANTISSA_MASK_F4) << 23
    x_lo_f32 = tl.cvt(tl.float32, sign_f4 | exp_f4 | mantissa_f4) * (2 ** (EXP_BIAS_BF16 - EXP_BIAS_F4))

    sign_f4 = (x_hi & SIGN_MASK_F4) << 27
    exp_f4 = (x_hi & EXP_MASK_F4) << 23
    mantissa_f4 = (x_hi & MANTISSA_MASK_F4) << 23
    x_hi_f32 = tl.cvt(tl.float32, sign_f4 | exp_f4 | mantissa_f4) * (2 ** (EXP_BIAS_BF16 - EXP_BIAS_F4))

    # Apply scale
    x_lo_f32 = x_lo_f32 * scale
    x_hi_f32 = x_hi_f32 * scale

    # Handle special cases
    x_lo_f32 = tl.where(x_lo_f32 == 0.0, ZERO_BITS_F32, x_lo_f32)
    x_lo_f32 = tl.where(x_lo_f32 < DENORMAL_THRESHOLD_F32, DENORMAL_THRESHOLD_F32, x_lo_f32)
    x_hi_f32 = tl.where(x_hi_f32 == 0.0, ZERO_BITS_F32, x_hi_f32)
    x_hi_f32 = tl.where(x_hi_f32 < DENORMAL_THRESHOLD_F32, DENORMAL_THRESHOLD_F32, x_hi_f32)

    # Convert to bf16
    x_lo_bf16 = tl.cvt(tl.bfloat16, x_lo_f32)
    x_hi_bf16 = tl.cvt(tl.bfloat16, x_hi_f32)

    # Store results
    tl.store(output_ptr + offsets, x_lo_bf16, mask=mask)
    tl.store(output_ptr + offsets + BLOCK_SIZE_IN // 2, x_hi_bf16, mask=mask)

import torch
import triton
import triton.language as tl

def triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size=1024):
    # Constants for conversion
    SIGN_MASK_F4 = 0x8
    EXP_MASK_F4 = 0x7
    MANTISSA_MASK_F4 = 0x0
    EXP_BIAS_F4 = -7
    EXP_BIAS_BF16 = -127
    SIGN_MASK_BF16 = 0x8000
    EXP_MASK_BF16 = 0x7F80
    MANTISSA_MASK_BF16 = 0x007F
    ZERO_BITS_F32 = 0x00000000
    DENORMAL_THRESHOLD_F32 = 0x00000001

    # Input and output tensors
    n_elements_in = x.numel() * 2  # Each uint8 contains 2 fp4 values
    output = torch.empty(n_elements_in, dtype=torch.bfloat16, device=x.device)

    # Grid and block sizes
    grid = (triton.cdiv(n_elements_in, mx_block_size),)

    # Launch kernel
    triton_f4_to_scaled_bf16_kernel[grid](
        x, s_e8m0, output, n_elements_in,
        BLOCK_SIZE_IN=mx_block_size,
        SIGN_MASK_F4=SIGN_MASK_F4,
        EXP_MASK_F4=EXP_MASK_F4,
        MANTISSA_MASK_F4=MANTISSA_MASK_F4,
        EXP_BIAS_F4=EXP_BIAS_F4,
        EXP_BIAS_BF16=EXP_BIAS_BF16,
        SIGN_MASK_BF16=SIGN_MASK_BF16,
        EXP_MASK_BF16=EXP_MASK_BF16,
        MANTISSA_MASK_BF16=MANTISSA_MASK_BF16,
        ZERO_BITS_F32=ZERO_BITS_F32,
        DENORMAL_THRESHOLD_F32=DENORMAL_THRESHOLD_F32
    )

    return output
