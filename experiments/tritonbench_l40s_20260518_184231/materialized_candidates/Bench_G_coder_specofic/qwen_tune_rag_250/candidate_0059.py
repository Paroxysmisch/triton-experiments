_high_bits)

    sign = x & sign_mask_f4
    x = x & mantissa_mask_f4

    exp_f4 = (x << (e8m0_exponent_bias - f4_e2m1_exp_bias)) + f8m0_exponent_bias
    exp_f4 += (exp_f4 == 0) * (x + 1)
    is_special_case = exp_f4 == 0
    x += is_special_case * (1 - (1 << (e8m0_exponent_bias - f4_e2m1_exp_bias)))
    exp_f32 = ((x + (1 << mbits_f4_e2m1)) >> mbits_f4_e2m1) - f32_exp_bias
    exp_f32 += (exp_f32 == 0) * ((x + (1 << mbits_f4_e2m1)) + 1)
    man_f32 = (x << (mbits_f32 - mbits_f4_e2m1)) + (x >> (mbits_f4_e2m1 - mbits_f32))
    man_f32 += (man_f32 == 0) * ((x << (mbits_f32 - mbits_f4_e2m1)) + (x > 0))

    zero_mask = (sign | exp_f4 | man_f32) == 0
    exp_f32 += ((exp_f32 == 0) & (man_f32 != 0)) * (1 - (1 << f32_exp_bias))
    exp_f32 += (exp_f32 == 0) * (1 - (1 << f32_exp_bias))
    is_inf_or_nan = exp_f32 == f32_exp_bias
    inf_or_nan_mask = is_inf_or_nan & (man_f32 != 0)
    exp_f32 += inf_or_nan_mask * (1 << f32_exp_bias)
    exp_f32 = tl.where(is_special_case, exp_f4, exp_f32)
    inf_or_nan_mask |= ((exp_f32 + 1) == 0) & ((exp_f4 + 1) == 0)
    zero_mask |= (man_f32 == 0) & inf_or_nan_mask

    inf_or_nan_bf16 = (
        (exp_f32.to(tl.uint32) << (32 - ebits_f32 - mbits_f32)) | (man_f32 << (32 - ebits_f32))
    ).to(tl.bfloat16)
    zero_bf16 = (zero_bits_f32 + ((sign + (sign << 7)) & (1 - (1 - (man_f32 + 1))))) | (
        zero_point_five_bits_f32 & (man_f32 + 1)
    )
    zero_bf16 = zero_bf16.to(tl.bfloat16)
    return tl.where(zero_mask, zero_bf16, inf_or_nan_bf16)

@triton.jit
def triton_f4_to_bf16_kernel(
    x_ptr,
    output_ptr,
    n_elements_in,
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
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_IN
    offsets = block_start + tl.arange(0, BLOCK_SIZE_IN)
    mask = offsets < n_elements_in
    x = tl.load(x_ptr + offsets, mask=mask)
    return _fp4_packed_to_bf16(
        x,
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

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    mx_block_size,
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
    e8m0_exponent_bias,
    e8m0_exponent_nan_val,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_IN
    offsets = block_start + tl.arange(0, BLOCK_SIZE_IN)
    x_mask = offsets < n_elements_in
    x = tl.load(x_ptr + offsets, mask=x_mask)
    s_mask = offsets < mx_block_size
    s = tl.load(s_ptr + (offsets // 4), mask=s_mask)
    s = (s >> ((offsets % 4) * 8)) & 0xFF
    s = (s << (32 - 8)) >> (32 - 16)
    return _fp4_packed_to_bf16(
        x,
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
    ) * s

def triton_f4_to_bf16(x, dtype):
    assert x.is_cuda and x.dtype == torch.uint8
    assert dtype == torch.bfloat16
    n_elements_out = x.numel() * 2
    output = torch.empty(n_elements_out, device=x.device, dtype=dtype)
    grid = lambda meta: (triton.cdiv(n_elements_out, meta["BLOCK_SIZE_IN"]),)
    triton_f4_to_bf16_kernel[grid](
        x,
        output,
        n_elements_out,
        SIGN_MASK_F4,
        MANTISSA_MASK_F4,
        MBITS_F4_E2M1,
        EBITS_F4_E2M1,
        F4_E2M1_EXP_BIAS,
        MBITS_F32,
        EBITS_F32,
        F32_EXP_BIAS,
        ZERO_BITS_F32,
        ZERO_POINT_FIVE_BITS_F32,
        BLOCK_SIZE_IN=1024,
    )
    return output

def triton_f4_to_scaled_bf16(x, s):
    assert x.is_cuda and x.dtype == torch.uint8
    assert s.is_cuda and s.dtype == torch.uint8
    n_elements_out = x.numel() * 2
    output = torch.empty(n_elements_out, device=x.device, dtype=torch.bfloat16)
    mx_block_size = 4 * triton.next_power_of_2(s.numel())
    grid = lambda meta: (triton.cdiv(n_elements_out, meta["BLOCK_SIZE_IN"]),)
    triton_f4_to_scaled_bf16_kernel[grid](
        x,
        s,
        output,
        n_elements_out,
        mx_block_size,
        SIGN_MASK_F4,
        MANTISSA_MASK_F4,
        MBITS_F4_E2M1,
        EBITS_F4_E2M1,
        F4_E2M1_EXP_BIAS,
        MBITS_F32,
        EBITS_F32,
        F32_EXP_BIAS,
        ZERO_BITS_F32,
        ZERO_POINT_FIVE_BITS_F32,
        E8M0_EXPONENT_BIAS,
        E8M0_EXPONENT_NAN_VAL,
        BLOCK_SIZE_IN=1024,
    )
    return output
