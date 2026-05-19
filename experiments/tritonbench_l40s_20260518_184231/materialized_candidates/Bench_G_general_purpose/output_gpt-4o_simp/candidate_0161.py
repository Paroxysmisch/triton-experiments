import triton
import triton.language as tl

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr, s_ptr, output_ptr, n_elements_in,
    SIGN_MASK_F4, ZERO_BITS_F32, EXPONENT_BIAS_F4, EXPONENT_BIAS_BF16,
    BLOCK_SIZE_IN, **meta
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)
    mask = offsets < n_elements_in

    # Load packed fp4 values
    packed_fp4 = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Unpack 4-bit values (assuming 2 values per byte)
    fp4_values = (packed_fp4 & 0xF, packed_fp4 >> 4)

    # Initialize bf16 output
    bf16_output = tl.zeros([BLOCK_SIZE_IN * 2], dtype=tl.float32)

    for i in range(2):
        # Extract sign, exponent, and mantissa
        sign = (fp4_values[i] & SIGN_MASK_F4) << 28
        exponent = ((fp4_values[i] >> 1) & 0x7) + EXPONENT_BIAS_BF16 - EXPONENT_BIAS_F4
        mantissa = (fp4_values[i] & 0x1) << 23

        # Combine into bf16 format
        bf16_value = sign | (exponent << 23) | mantissa

        # Handle special cases (zero, denormal)
        is_zero = fp4_values[i] == 0
        bf16_value = tl.where(is_zero, ZERO_BITS_F32, bf16_value)

        # Convert to float32 for scaling
        bf16_value_f32 = tl.bitcast(bf16_value, tl.float32)

        # Load scale and apply
        scale = tl.load(s_ptr)
        scaled_value = bf16_value_f32 * scale

        # Store the result
        bf16_output[i::2] = scaled_value

    # Store the result
    tl.store(output_ptr + offsets * 2, bf16_output, mask=mask)

import torch

def triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size):
    assert x.dtype == torch.uint8, "Input tensor must be of type uint8 for packed fp4 values."
    assert s_e8m0.dtype == torch.float32, "Scale tensor must be of type float32."

    n_elements_in = x.numel()
    n_elements_out = n_elements_in * 2

    # Allocate output tensor
    output = torch.empty(n_elements_out, dtype=torch.bfloat16, device=x.device)

    # Constants
    SIGN_MASK_F4 = 0x8
    ZERO_BITS_F32 = 0x00000000
    EXPONENT_BIAS_F4 = 7
    EXPONENT_BIAS_BF16 = 127
    BLOCK_SIZE_IN = mx_block_size

    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements_in, BLOCK_SIZE_IN),)
    triton_f4_to_scaled_bf16_kernel[grid](
        x, s_e8m0, output, n_elements_in,
        SIGN_MASK_F4, ZERO_BITS_F32, EXPONENT_BIAS_F4, EXPONENT_BIAS_BF16,
        BLOCK_SIZE_IN
    )

    return output
