import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

# Constants for FP4 conversion
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x1
MBITS_F4_E2M1, EBITS_F4_E2M1 = 1, 2
F4_E2M1_EXP_BIAS = 7
MBITS_F32, EBITS_F32 = 23, 8
F32_EXP_BIAS = 127
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

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
    x_low = x_packed >> 4
    x_high = x_packed & 0xF
    x = tl.interleave(x_low, x_high)
    
    # Extract components
    sign = x & sign_mask_f4
    x_pos = x ^ sign
    zero_mask = x_pos == 0
    denorm_mask = x_pos == 1
    
    # Exponent handling
    exp_biased_f4 = x_pos >> mbits_f4_e2m1
    exp_f32 = (exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias).to(tl.int32) << mbits_f32
    
    # Mantissa handling
    mantissa_f4 = x_pos & mantissa_mask_f4
    mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
    
    # Combine components
    result = exp_f32 | mantissa_f32
    result = tl.where(zero_mask, zero_bits_f32, result)
    result = tl.where(denorm_mask, zero_point_five_bits_f32, result)
    sign_f32 = sign.to(tl.int32) << (mbits_f32 + ebits_f32 - mbits_f4_e2m1 - ebits_f4_e2m1)
    result |= sign_f32
    
    return result.to(tl.float32, bitcast=True).to(tl.bfloat16)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_IN": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE_IN": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE_IN": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE_IN": 1024}, num_warps=8),
    ],
    key=["n_elements_in"],
)
@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    mx_block_size: tl.constexpr,
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
    e8m0_exponent_bias: tl.constexpr,
    e8m0_exponent_nan_val: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE_IN
    offsets = block_start + tl.arange(0, BLOCK_SIZE_IN)
    mask = offsets < n_elements_in
    
    # Load and convert FP4 data
    x_packed = tl.load(x_ptr + offsets, mask=mask)
    bf16_vals = _fp4_packed_to_bf16(
        x_packed,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
    )
    
    # Load and process scaling factors
    scale_block = pid * (BLOCK_SIZE_IN // 16)
    scale_offsets = scale_block + tl.arange(0, BLOCK_SIZE_IN // 16)
    scales = tl.load(s_ptr + scale_offsets, mask=scale_offsets < (n_elements_in // 16))
    scale_exp = scales.to(tl.int16) - e8m0_exponent_bias
    scale_factors = tl.where(
        scales != e8m0_exponent_nan_val,
        libdevice.pow(2.0, scale_exp),
        float('nan')
    ).to(tl.bfloat16)
    
    # Apply scaling
    scaled_vals = bf16_vals * tl.reshape(scale_factors, (-1, 1))
    
    # Store results
    output_offsets = pid * BLOCK_SIZE_IN * 2 + tl.arange(0, BLOCK_SIZE_IN * 2)
    tl.store(output_ptr + output_offsets, scaled_vals, mask=output_offsets < (n_elements_in * 2))

def triton_f4_to_scaled_bf16(
    x: torch.Tensor,
    s_e8m0: torch.Tensor,
    mx_block_size: int = 32
) -> torch.Tensor:
    # Validate inputs
    assert x.is_cuda and s_e8m0.is_cuda
    assert x.is_contiguous()
    assert x.dtype == torch.uint8
    
    # Prepare output tensor
    output_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(output_shape, device=x.device, dtype=torch.bfloat16)
    
    # Launch kernel
    n_elements_in = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),)
    
    triton_f4_to_scaled_bf16_kernel[grid](
        x_ptr=x,
        s_ptr=s_e8m0,
        output_ptr=output,
        n_elements_in=n_elements_in,
        mx_block_size=mx_block_size,
        sign_mask_f4=SIGN_MASK_F4,
        mantissa_mask_f4=MANTISSA_MASK_F4,
        mbits_f4_e2m1=MBITS_F4_E2M1,
        ebits_f4_e2m1=EBITS_F4_E2M1,
        f4_e2m1_exp_bias=F4_E2M1_EXP_BIAS,
        mbits_f32=MBITS_F32,
        ebits_f32=EBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
        zero_bits_f32=ZERO_BITS_F32,
        zero_point_five_bits_f32=ZERO_POINT_FIVE_BITS_F32,
        e8m0_exponent_bias=E8M0_EXPONENT_BIAS,
        e8m0_exponent_nan_val=E8M0_EXPONENT_NAN_VAL,
        BLOCK_SIZE_IN=512,
    )
    
    return output
