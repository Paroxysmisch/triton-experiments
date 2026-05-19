1
    normal_mask = ~zero_mask & ~denormal_mask

    sign_bf16 = (sign_f4 << (MBITS_F32 - mbits_f4_e2m1)) | (sign_f4 << (EBITS_F32 - 1))
    sign_bf16 = tl.where(denormal_mask, sign_bf16 | 0x40000000, sign_bf16)

    mantissa_f4 = x & mantissa_mask_f4
    mantissa_f4 = (mantissa_f4 + 1) << (MBITS_F32 - mbits_f4_e2m1)

    normal_bf16 = sign_bf16 | mantissa_f4 | 0x40000000
    normal_bf16 = normal_bf16 + (normal_bf16 == 0x40000000)

    zero_bf16 = zero_bits_f32
    zero_bf16 = tl.where(denormal_mask, zero_point_five_bits_f32, zero_bf16)

    result = tl.where(zero_mask, zero_bf16, normal_bf16)
    result = result.to(tl.bfloat16)
    return result

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
    result = _fp4_packed_to_bf16(
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
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    mx_block_size: tl.constexpr,
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
    x_ptr_offset = x_ptr + offsets
    s_ptr_offset = s_ptr + offsets // mx_block_size
    x = tl.load(x_ptr_offset, mask=x_mask)
    s = tl.load(s_ptr_offset, mask=x_mask)

    s_bf16 = _fp4_packed_to_bf16(
        s,
        e8m0_sign_mask,
        e8m0_mantissa_mask,
        e8m0_mbits,
        e8m0_ebits,
        e8m0_exp_bias,
        f32_mbits,
        f32_ebits,
        f32_exp_bias,
        f32_zero_bits,
        f32_zero_point_five_bits,
    )

    s_f32 = s_bf16.to(tl.float32)
    s_f32 = tl.where(tl.abs(s_f32) <= 1.0, s_f32, 1.0)

    result = x.to(tl.float32) * s_f32
    result = _fp4_packed_to_bf16(
        result,
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
    tl.store(output_ptr + offsets, result, mask=x_mask)

def triton_f4_to_bf16(x, *, dtype=None, max_block_size=None):
    if dtype is None:
        dtype = torch.bfloat16
    if max_block_size is None:
        max_block_size = 1024 * 1024
    assert x.is_contiguous() and x.is_cuda
    output = torch.empty(x.numel(), device=x.device, dtype=dtype)
    grid_fn = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE_IN"]),)
    with torch.cuda.device(x.device):
        triton.jit._triton_compile(
            triton_f4_to_bf16_kernel,
            triton.FinalizeMe(
                {
                    "x_ptr": x,
                    "output_ptr": output,
                    "n_elements_in": x.numel(),
                    "sign_mask_f4": 0x8,
                    "mantissa_mask_f4": 0x1,
                    "mbits_f4_e2m1": 1,
                    "ebits_f4_e2m1": 2,
                    "f4_e2m1_exp_bias": 7,
                    "mbits_f32": 23,
                    "ebits_f32": 8,
                    "f32_exp_bias": 127,
                    "zero_bits_f32": 0x0,
                    "zero_point_five_bits_f32": 0x3F000000,
                }
            ),
            signature={0: "*fp4_packed", 1: "*bf16"},
            device=x.device.index,
            grid=grid_fn,
        )
    return output

def triton_f4_to_scaled_bf16(x, s_e8m0):
    assert x.is_contiguous() and x.is_cuda
    assert s_e8m0.is_contiguous() and s_e8m0.is_cuda
    output = torch.empty(x.numel(), device=x.device, dtype=torch.bfloat16)
    grid_fn = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE_IN"]),)
    with torch.cuda.device(x.device):
        triton.jit._triton_compile(
            triton_f4_to_scaled_bf16_kernel,
            triton.FinalizeMe(
                {
                    "x_ptr": x,
                    "s_ptr": s_e8m0,
                    "output_ptr": output,
                    "n_elements_in": x.numel(),
                    "mx_block_size": 16,
                    "sign_mask_f4": 0x8,
                    "mantissa_mask_f4": 0x1,
                    "mbits_f4_e2m1": 1,
                    "ebits_f4_e2m1": 2,
                    "f4_e2m1_exp_bias": 7,
                    "mbits_f32": 23,
                    "ebits_f32": 8,
                    "f32_exp_bias": 127,
                    "zero_bits_f32": 0x0,
                    "zero_point_five_bits_f32": 0x3F000000,
                    "e8m0_exponent_bias": 127,
                    "e8m0_exponent_nan_val": 255,
                }
            ),
            signature={0: "*fp4_packed", 1: "*e8m0", 2: "*bf16"},
            device=x.device.index,
            grid=grid_fn,
        )
    return output
