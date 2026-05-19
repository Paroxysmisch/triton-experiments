import triton
import triton.language as tl
import torch

# Constants and bit masks definitions
SIGN_MASK_F4             = 0b1000  # sign bit: 1 bit
MANTISSA_MASK_F4         = 0b0011  # mantissa bits: 2 bits
ZERO_BITS_F32            = 0x00000000
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000  # 0.5 in FP32, used for denormal handling
# Exponent + mantissa sizes (used for shifting)
EBITS_F4_E2M1  = 3      # for shifting exponent left by 2 + mantissa left by 1 to form partial 32-bit pattern
MBITS_F4_E2M1  = 1
EBITS_F32      = 23
MBITS_F32      = 0
# Exponent offset (difference in bias: BF16 has bias 127, FP4 has bias 7)
EXP_BIAS_OFFSET = 127 - 7

@triton.jit
def _fp4_packed_to_bf16(in_data, idx):
    # Extract the low 4 bits and high 4 bits
    fp4_val = tl.load(in_data + idx, mask=True, other=0, eviction_policy="evict_last")
    low4  = fp4_val & 0xF
    high4 = (fp4_val >> 4) & 0xF

    # Convert low4
    sign_low  = (low4 & SIGN_MASK_F4) >> 3
    exp_low   = (low4 >> 1) & 0b11
    mant_low  = (low4 & MANTISSA_MASK_F4)
    # Convert high4
    sign_high = (high4 & SIGN_MASK_F4) >> 3
    exp_high  = (high4 >> 1) & 0b11
    mant_high = (high4 & MANTISSA_MASK_F4)

    # Reconstruct FP32 bits for low4
    sign_bits_low  = sign_low << 31
    # If exponent == 0 and mantissa != 0, it's a denormal in FP4 => set exponent to 0, mantissa ~ 0.5
    # If exponent == 0 and mantissa == 0 => zero
    is_denorm_low = (exp_low == 0) & (mant_low != 0)
    is_zero_low   = (exp_low == 0) & (mant_low == 0)
    exp_bits_low  = tl.where(is_denorm_low, tl.full([1], (EXP_BIAS_OFFSET << EBITS_F32), tl.int32),
                                          tl.where(is_zero_low, tl.full([1], 0, tl.int32),
                                                    (exp_low + EXP_BIAS_OFFSET) << EBITS_F32))
    mant_bits_low = mant_low << (EBITS_F32 - MBITS_F4_E2M1)
    # Denormal => approximate mantissa bits for a 0.5 pattern
    denorm_or_zero_low = tl.where(is_denorm_low, ZERO_POINT_FIVE_BITS_F32, ZERO_BITS_F32)
    low_fp32_bits = tl.where(is_denorm_low | is_zero_low, denorm_or_zero_low, sign_bits_low | exp_bits_low | mant_bits_low)

    # Reconstruct FP32 bits for high4
    sign_bits_high = sign_high << 31
    is_denorm_high = (exp_high == 0) & (mant_high != 0)
    is_zero_high   = (exp_high == 0) & (mant_high == 0)
    exp_bits_high  = tl.where(is_denorm_high, tl.full([1], (EXP_BIAS_OFFSET << EBITS_F32), tl.int32),
                                          tl.where(is_zero_high, tl.full([1], 0, tl.int32),
                                                    (exp_high + EXP_BIAS_OFFSET) << EBITS_F32))
    mant_bits_high = mant_high << (EBITS_F32 - MBITS_F4_E2M1)
    denorm_or_zero_high = tl.where(is_denorm_high, ZERO_POINT_FIVE_BITS_F32, ZERO_BITS_F32)
    high_fp32_bits = tl.where(is_denorm_high | is_zero_high, denorm_or_zero_high,
                              sign_bits_high | exp_bits_high | mant_bits_high)

    # Convert reconstructed FP32 to BF16: shift-right 16 bits
    low_bf16  = (low_fp32_bits  >> 16) & 0xFFFF
    high_bf16 = (high_fp32_bits >> 16) & 0xFFFF

    return (low_bf16, high_bf16)

@triton.jit
def triton_f4_to_bf16_kernel(in_data_ptr, out_data_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Each offset processes one byte => convert 2 FP4 at once
    bf16_low, bf16_high = _fp4_packed_to_bf16(in_data_ptr, offsets)
    # Write out results
    out_idx_low  = start * 2  + tl.arange(0, BLOCK_SIZE) * 2
    out_idx_high = out_idx_low + 1

    tl.store(out_data_ptr + out_idx_low,  bf16_low,  mask=mask)
    tl.store(out_data_ptr + out_idx_high, bf16_high, mask=mask)

def triton_f4_to_bf16(input_fp4: torch.Tensor):
    assert input_fp4.dtype == torch.uint8, "Input must be packed FP4 in uint8."
    input_fp4 = input_fp4.contiguous()
    n_elements = input_fp4.numel()
    # Each byte => 2 FP4 => result tensor has 2*N shape in BF16
    out_shape = [n_elements * 2]
    out_bf16  = torch.empty(out_shape, dtype=torch.bfloat16, device=input_fp4.device)

    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    triton_f4_to_bf16_kernel[grid](
        input_fp4, out_bf16, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out_bf16
