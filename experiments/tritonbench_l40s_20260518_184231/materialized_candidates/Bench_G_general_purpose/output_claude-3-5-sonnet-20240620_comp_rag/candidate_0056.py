import torch
import triton
import triton.language as tl

# Constants
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x1
MBITS_F4_E2M1, EBITS_F4_E2M1 = 1, 2
MBITS_F32, EBITS_F32 = 23, 8
F4_E2M1_EXP_BIAS = 7
F32_EXP_BIAS = 127
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000

@triton.jit
def _fp4_packed_to_bf16(x_packed, sign_mask_f4, mantissa_mask_f4, mbits_f4_e2m1, 
                        ebits_f4_e2m1, f4_e2m1_exp_bias, mbits_f32, ebits_f32,
                        f32_exp_bias, zero_bits_f32, zero_point_five_bits_f32):
    # Extract low and high 4 bits
    x_low = x_packed >> 4  
    x_high = x_packed & 0xF
    x = tl.interleave(x_low, x_high)

    # Extract sign bit and handle special cases
    sign = x & sign_mask_f4
    x_abs = x ^ sign
    
    is_zero = x_abs == 0
    is_denormal = x_abs == 1

    # Extract and adjust exponent
    exp_f4 = x_abs >> mbits_f4_e2m1
    exp_f32 = exp_f4 - f4_e2m1_exp_bias + f32_exp_bias
    exp_f32 = exp_f32.to(tl.int32) << mbits_f32

    # Extract and adjust mantissa 
    mantissa = x_abs & mantissa_mask_f4
    mantissa_f32 = mantissa.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)

    # Combine components
    result = exp_f32 | mantissa_f32
    
    # Handle special cases
    result = tl.where(is_zero, zero_bits_f32, result)
    result = tl.where(is_denormal, zero_point_five_bits_f32, result)

    # Add sign bit
    sign_f32 = sign.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1)
    result = result | sign_f32

    # Convert to bf16
    return result.to(tl.float32, bitcast=True).to(tl.bfloat16)

@triton.jit
def triton_f4_to_bf16_kernel(x_ptr, output_ptr, n_elements_in,
                            sign_mask_f4: tl.constexpr, mantissa_mask_f4: tl.constexpr,
                            mbits_f4_e2m1: tl.constexpr, ebits_f4_e2m1: tl.constexpr,
                            f4_e2m1_exp_bias: tl.constexpr, mbits_f32: tl.constexpr,
                            ebits_f32: tl.constexpr, f32_exp_bias: tl.constexpr,
                            zero_bits_f32: tl.constexpr, zero_point_five_bits_f32: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    # Calculate indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_in

    # Load and convert values
    x = tl.load(x_ptr + offsets, mask=mask)
    output = _fp4_packed_to_bf16(x, sign_mask_f4, mantissa_mask_f4, mbits_f4_e2m1,
                                ebits_f4_e2m1, f4_e2m1_exp_bias, mbits_f32, ebits_f32,
                                f32_exp_bias, zero_bits_f32, zero_point_five_bits_f32)

    # Store results
    output_offsets = block_start * 2 + tl.arange(0, BLOCK_SIZE * 2)
    output_mask = output_offsets < (n_elements_in * 2)
    tl.store(output_ptr + output_offsets, output, mask=output_mask)

def triton_f4_to_bf16(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous()
    n_elements = x.numel()
    output = torch.empty(n_elements * 2, device=x.device, dtype=torch.bfloat16)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    triton_f4_to_bf16_kernel[grid](
        x, output, n_elements,
        SIGN_MASK_F4, MANTISSA_MASK_F4,
        MBITS_F4_E2M1, EBITS_F4_E2M1,
        F4_E2M1_EXP_BIAS, MBITS_F32,
        EBITS_F32, F32_EXP_BIAS,
        ZERO_BITS_F32, ZERO_POINT_FIVE_BITS_F32,
        BLOCK_SIZE
    )
    
    return output.view(*x.shape[:-1], -1)
