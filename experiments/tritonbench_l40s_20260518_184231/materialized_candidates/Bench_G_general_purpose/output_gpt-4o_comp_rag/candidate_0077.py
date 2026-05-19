import torch
import triton
import triton.language as tl
from torch.utils._triton import has_triton
from torch._inductor.runtime.triton_helpers import libdevice

# Constants for masks and biases
SIGN_MASK_F4 = 0x8  # 1000
MANTISSA_MASK_F4 = 0x1  # 0001
MBITS_F4_E2M1, EBITS_F4_E2M1 = 1, 2
MBITS_F32, EBITS_F32 = 23, 8
F4_E2M1_EXP_BIAS = 7
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000

@triton.jit
def _fp4_packed_to_bf16(
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
):
    # Extract low and high bits
    x_low_bits = x_packed >> 4
    x_high_bits = x_packed & 0xF
    x = tl.interleave(x_low_bits, x_high_bits)

    # Extract sign and absolute value
    sign_f4 = x & sign_mask_f4
    x_pos = x ^ sign_f4

    # Identify special cases: zero and denormal (0.5)
    zero_mask = x_pos == 0
    denormal_mask = x_pos == 1

    # Calculate biased exponent for FP32
    exp_biased_f4 = x_pos >> mbits_f4_e2m1
    exp_biased_f32 = exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias
    exp_biased_f32 = exp_biased_f32.to(tl.int32) << mbits_f32

    # Calculate mantissa for FP32
    mantissa_f4 = x_pos & mantissa_mask_f4
    mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
    output = mantissa_f32

    # Combine exponent and mantissa
    result = exp_biased_f32 | mantissa_f32
    result = tl.where(zero_mask, zero_bits_f32, result)
    result = tl.where(denormal_mask, zero_point_five_bits_f32, result)

    # Add sign bit
    sign_f32 = sign_f4.to(tl.int32) << (
        mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1
    )
    result = result | sign_f32

    # Convert to FP32 and then to BF16
    output = result.to(tl.float32, bitcast=True)
    output = output.to(tl.bfloat16)
    return output

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
    # Determine program ID and element positions
    pid = tl.program_id(axis=0)
    n_elements_out = n_elements_in * 2
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

    block_start_in = pid * BLOCK_SIZE_IN
    offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN)

    mask_in = offsets_in < n_elements_in

    # Load packed FP4 values
    x_packed = tl.load(x_ptr + offsets_in, mask=mask_in)
    output = _fp4_packed_to_bf16(
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

    block_start_out = pid * BLOCK_SIZE_OUT
    offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT)
    mask_out = offsets_out < n_elements_out

    # Store the converted BF16 values
    tl.store(output_ptr + offsets_out, output, mask=mask_out)

def triton_f4_to_bf16(x: torch.Tensor):
    new_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda
    n_elements_in = x.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),
    )
    triton_f4_to_bf16_kernel[grid](
        x,
        output,
        n_elements_in,
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
        BLOCK_SIZE_IN=512,
    )
    return output
