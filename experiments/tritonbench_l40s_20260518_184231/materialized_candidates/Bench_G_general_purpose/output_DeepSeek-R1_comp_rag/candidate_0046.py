import torch
import triton
import triton.language as tl

# Constants for FP4 and FP32 configurations
SIGN_MASK_F4 = 0x8  # 1000 (4-bit sign mask)
MANTISSA_MASK_F4 = 0x1  # 0001 (4-bit mantissa mask)
MBITS_F4_E2M1, EBITS_F4_E2M1 = 1, 2  # FP4 (E2M1) format parameters
MBITS_F32, EBITS_F32 = 23, 8  # FP32 format parameters
F4_E2M1_EXP_BIAS = 7  # Exponent bias for FP4
ZERO_BITS_F32 = 0x0  # Zero in FP32
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000  # 0.5 in FP32

@triton.jit
def _fp4_packed_to_bf16(
    x_packed,
    sign_mask_f4: tl.constexpr,
    mantissa_mask_f4: tl.constexpr,
    mbits_f4_e2m1: tl.constexpr,
    ebits_f4_e2m1: tl.constexpr,
    f4_e2m1_exp_bias: tl.constexpr,
    mbits_f32: tl.constexpr,
    ebits_f32: tl.constexpr,
    f32_exp_bias: tl.constexpr,
    zero_bits_f32: tl.constexpr,
    zero_point_five_bits_f32: tl.constexpr,
):
    # Unpack and interleave 4-bit values
    x_low_bits = x_packed >> 4
    x_high_bits = x_packed & 0xF
    x = tl.interleave(x_low_bits, x_high_bits)
    
    # Extract sign and absolute value
    sign_f4 = x & sign_mask_f4
    x_pos = x ^ sign_f4
    
    # Identify special cases
    zero_mask = x_pos == 0
    denormal_mask = x_pos == 1
    
    # Exponent conversion
    exp_biased_f4 = x_pos >> mbits_f4_e2m1
    exp_biased_f32 = (exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias).to(tl.int32) << mbits_f32
    
    # Mantissa conversion
    mantissa_f4 = x_pos & mantissa_mask_f4
    mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
    
    # Combine components
    result = exp_biased_f32 | mantissa_f32
    result = tl.where(zero_mask, zero_bits_f32, result)
    result = tl.where(denormal_mask, zero_point_five_bits_f32, result)
    
    # Add sign bit
    sign_f32 = sign_f4.to(tl.int32) << (mbits_f32 + ebits_f32 - mbits_f4_e2m1 - ebits_f4_e2m1)
    result |= sign_f32
    
    # Convert to bfloat16
    return result.to(tl.float32, bitcast=True).to(tl.bfloat16)

@triton.jit
def triton_f4_to_bf16_kernel(
    x_ptr,
    output_ptr,
    n_elements_in,
    sign_mask_f4: tl.constexpr,
    mantissa_mask_f4: tl.constexpr,
    mbits_f4_e2m1: tl.constexpr,
    ebits_f4_e2m1: tl.constexpr,
    f4_e2m1_exp_bias: tl.constexpr,
    mbits_f32: tl.constexpr,
    ebits_f32: tl.constexpr,
    f32_exp_bias: tl.constexpr,
    zero_bits_f32: tl.constexpr,
    zero_point_five_bits_f32: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE_IN
    offsets = block_start + tl.arange(0, BLOCK_SIZE_IN)
    
    mask = offsets < n_elements_in
    x_packed = tl.load(x_ptr + offsets, mask=mask)
    
    # Process and store
    bf16_vals = _fp4_packed_to_bf16(
        x_packed, sign_mask_f4, mantissa_mask_f4,
        mbits_f4_e2m1, ebits_f4_e2m1, f4_e2m1_exp_bias,
        mbits_f32, ebits_f32, f32_exp_bias,
        zero_bits_f32, zero_point_five_bits_f32
    )
    
    output_offset = pid * BLOCK_SIZE_IN * 2
    output_indices = output_offset + tl.arange(0, BLOCK_SIZE_IN * 2)
    tl.store(output_ptr + output_indices, bf16_vals, mask=output_indices < n_elements_in * 2)

def triton_f4_to_bf16(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((*x.shape[:-1], x.shape[-1] * 2), 
                        device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous() and x.is_cuda
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE_IN']),)
    
    triton_f4_to_bf16_kernel[grid](
        x, output, n_elements,
        sign_mask_f4=SIGN_MASK_F4,
        mantissa_mask_f4=MANTISSA_MASK_F4,
        mbits_f4_e2m1=MBITS_F4_E2M1,
        ebits_f4_e2m1=EBITS_F4_E2M1,
        f4_e2m1_exp_bias=F4_E2M1_EXP_BIAS,
        mbits_f32=MBITS_F32,
        ebits_f32=EBITS_F32,
        f32_exp_bias=127,  # Standard FP32 exponent bias
        zero_bits_f32=ZERO_BITS_F32,
        zero_point_five_bits_f32=ZERO_POINT_FIVE_BITS_F32,
        BLOCK_SIZE_IN=1024  # Optimized block size
    )
    return output
