import torch
import triton
import triton.language as tl
from torch.utils._triton import has_triton
from torch._inductor.runtime.triton_helpers import libdevice

# Constants for fp4 and fp32 formats
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x1
MBITS_F4_E2M1, EBITS_F4_E2M1 = 1, 2
MBITS_F32, EBITS_F32 = 23, 8
F4_E2M1_EXP_BIAS = 7
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

@triton.jit
def _fp4_to_bf16_conversion(x_packed):
    # Extract lower and upper 4 bits
    x_low = x_packed >> 4
    x_high = x_packed & 0xF
    x = tl.interleave(x_low, x_high)
    
    # Extract sign and handle special cases
    sign = x & SIGN_MASK_F4
    x_abs = x ^ sign
    
    is_zero = x_abs == 0
    is_denormal = x_abs == 1
    
    # Convert exponent
    exp = x_abs >> MBITS_F4_E2M1
    exp_f32 = (exp - F4_E2M1_EXP_BIAS + F32_EXP_BIAS).to(tl.int32) << MBITS_F32
    
    # Convert mantissa
    mantissa = x_abs & MANTISSA_MASK_F4
    mantissa_f32 = mantissa.to(tl.int32) << (MBITS_F32 - MBITS_F4_E2M1)
    
    # Combine components
    result = exp_f32 | mantissa_f32
    result = tl.where(is_zero, ZERO_BITS_F32, result)
    result = tl.where(is_denormal, ZERO_POINT_FIVE_BITS_F32, result)
    
    # Apply sign
    sign_f32 = sign.to(tl.int32) << (MBITS_F32 + EBITS_F32 - EBITS_F4_E2M1 - MBITS_F4_E2M1)
    result = result | sign_f32
    
    # Convert to bf16
    return result.to(tl.float32, bitcast=True).to(tl.bfloat16)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=['n_elements']
)
@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr, s_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute linear index and load data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load packed fp4 values
    x_packed = tl.load(x_ptr + offsets, mask=mask)
    
    # Convert to bf16
    values = _fp4_to_bf16_conversion(x_packed)
    
    # Load and apply scaling factor
    scale_idx = offsets // 32  # One scale per 32 elements
    scale_mask = scale_idx < (n_elements + 31) // 32
    scale = tl.load(s_ptr + scale_idx, mask=scale_mask)
    
    # Convert e8m0 scale to fp32
    scale_exp = scale.to(tl.int32) - E8M0_EXPONENT_BIAS
    scale_fp = libdevice.pow(2.0, scale_exp).to(tl.bfloat16)
    scale_fp = tl.where(scale != E8M0_EXPONENT_NAN_VAL, scale_fp, float('nan'))
    
    # Apply scaling
    output = values * scale_fp
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_f4_to_scaled_bf16(x: torch.Tensor, s_e8m0: torch.Tensor):
    assert x.is_cuda and x.is_contiguous()
    n_elements = x.numel()
    
    # Create output tensor
    output = torch.empty(n_elements * 2, 
                        device=x.device,
                        dtype=torch.bfloat16)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    triton_f4_to_scaled_bf16_kernel[grid](
        x, s_e8m0, output,
        n_elements,
    )
    
    return output.reshape(*x.shape[:-1], -1)
