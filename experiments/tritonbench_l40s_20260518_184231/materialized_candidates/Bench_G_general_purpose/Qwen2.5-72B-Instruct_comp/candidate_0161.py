import triton
import triton.language as tl

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr, s_ptr, output_ptr, n_elements_in,
    SIGN_MASK_F4: tl.constexpr, EXP_MASK_F4: tl.constexpr, MANTISSA_MASK_F4: tl.constexpr,
    SIGN_MASK_BF16: tl.constexpr, EXP_MASK_BF16: tl.constexpr, MANTISSA_MASK_BF16: tl.constexpr,
    EXP_BIAS_F4: tl.constexpr, EXP_BIAS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the packed 4-bit floats and scaling factors
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements_in, other=0)
    s = tl.load(s_ptr + (offsets // 2), mask=offsets < n_elements_in, other=0)

    # Decompose each packed byte into two 4-bit numbers
    x0 = (x & 0x0F)
    x1 = (x >> 4)

    # Extract sign, exponent, and mantissa bits
    sign0 = (x0 & SIGN_MASK_F4) << 15
    sign1 = (x1 & SIGN_MASK_F4) << 15
    exp0 = (x0 & EXP_MASK_F4) << 10
    exp1 = (x1 & EXP_MASK_F4) << 10
    mant0 = (x0 & MANTISSA_MASK_F4) << 7
    mant1 = (x1 & MANTISSA_MASK_F4) << 7

    # Apply scaling factor
    exp0 = (exp0 + (s & 0x7F) - EXP_BIAS_F4 + EXP_BIAS_BF16) & EXP_MASK_BF16
    exp1 = (exp1 + (s & 0x7F) - EXP_BIAS_F4 + EXP_BIAS_BF16) & EXP_MASK_BF16

    # Combine sign, exponent, and mantissa to form bf16
    bf16_0 = sign0 | exp0 | mant0
    bf16_1 = sign1 | exp1 | mant1

    # Handle special cases (zero and denormals)
    bf16_0 = tl.where((x0 == 0), 0, bf16_0)
    bf16_1 = tl.where((x1 == 0), 0, bf16_1)

    # Store the results
    output_offsets = offsets * 2
    tl.store(output_ptr + output_offsets, bf16_0, mask=offsets < n_elements_in)
    tl.store(output_ptr + output_offsets + 1, bf16_1, mask=offsets < n_elements_in)

import torch
import triton
import triton.language as tl

def triton_f4_to_scaled_bf16(x: torch.Tensor, s_e8m0: torch.Tensor, mx_block_size: int):
    # Ensure input tensors are contiguous and on the CUDA device
    x = x.contiguous().cuda()
    s_e8m0 = s_e8m0.contiguous().cuda()

    # Determine the shape and size of the output tensor
    n_elements_in = x.numel()
    output = torch.empty(n_elements_in * 2, dtype=torch.bfloat16, device=x.device)

    # Compute grid configuration
    grid = (triton.cdiv(n_elements_in, mx_block_size),)

    # Define constant expressions
    SIGN_MASK_F4 = 0x08
    EXP_MASK_F4 = 0x07
    MANTISSA_MASK_F4 = 0x00
    SIGN_MASK_BF16 = 0x8000
    EXP_MASK_BF16 = 0x7F80
    MANTISSA_MASK_BF16 = 0x007F
    EXP_BIAS_F4 = 7
    EXP_BIAS_BF16 = 127

    # Launch the Triton kernel
    triton_f4_to_scaled_bf16_kernel[grid](
        x, s_e8m0, output, n_elements_in,
        SIGN_MASK_F4, EXP_MASK_F4, MANTISSA_MASK_F4,
        SIGN_MASK_BF16, EXP_MASK_BF16, MANTISSA_MASK_BF16,
        EXP_BIAS_F4, EXP_BIAS_BF16,
        mx_block_size
    )

    return output
