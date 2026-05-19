import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements_in']
)
@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    SIGN_MASK_F4: tl.constexpr,      # 0x8 (binary 1000)
    EXP_MASK_F4: tl.constexpr,       # 0x6 (binary 0110)
    EXP_SHIFT_F4: tl.constexpr,      # 1 bit right shift
    EXP_BIAS_F4: tl.constexpr,       # 3 for 2 exponent bits
    MANTISSA_MASK_F4: tl.constexpr,  # 0x1 (binary 0001)
    MANTISSA_BITS_F4: tl.constexpr,  # 1 mantissa bit
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_in

    # Load packed bytes (n_elements_in)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Unpack 4-bit values
    elem_hi = (x >> 4) & 0x0F
    elem_lo = x & 0x0F
    elements = tl.join(elem_hi, elem_lo)  # Creates 2*BLOCK_SIZE elements

    # Prepare output indices and scaling factors
    output_offsets = 2 * block_start + tl.arange(0, 2 * BLOCK_SIZE)
    output_mask = output_offsets < (2 * n_elements_in)
    scales = tl.load(s_ptr + output_offsets, mask=output_mask)

    # Process elements
    sign = (elements & SIGN_MASK_F4) != 0
    exp = (elements & EXP_MASK_F4) >> EXP_SHIFT_F4
    mantissa = elements & MANTISSA_MASK_F4

    # Handle special cases
    is_zero = (exp == 0) & (mantissa == 0)
    exp_unbiased = tl.where(exp == 0, 0, exp - EXP_BIAS_F4)
    
    # Apply scaling and convert to bf16
    scaled_exp = exp_unbiased + scales
    exp_bf16 = tl.where(is_zero, 0, scaled_exp + 127)  # BF16 bias: 127
    exp_bf16 = tl.minimum(tl.maximum(exp_bf16, 0), 255)

    # Expand mantissa to 7 bits
    mantissa_bf16 = mantissa << (7 - MANTISSA_BITS_F4)
    
    # Combine components
    bf16_bits = (
        (sign.to(tl.uint16) << 15) |
        (exp_bf16.to(tl.uint16) << 7) |
        mantissa_bf16.to(tl.uint16)
    )
    result = tl.bitcast(bf16_bits, tl.bfloat16)

    # Store results
    tl.store(output_ptr + output_offsets, result, mask=output_mask)

def triton_f4_to_scaled_bf16(
    x: torch.Tensor,
    s_e8m0: torch.Tensor,
    mx_block_size: int = 1024
) -> torch.Tensor:
    # Ensure input compatibility
    assert x.dtype == torch.uint8, "Input must be packed uint8"
    assert x.is_cuda and x.is_contiguous()
    assert s_e8m0.is_cuda and s_e8m0.is_contiguous()
    assert 2 * x.numel() == s_e8m0.numel(), "Scaling factors must match output size"

    # Create output tensor
    output = torch.empty_like(s_e8m0, dtype=torch.bfloat16)

    # Kernel configuration
    n_elements_in = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements_in, meta['BLOCK_SIZE']),)

    # Launch kernel
    triton_f4_to_scaled_bf16_kernel[grid](
        x_ptr=x,
        s_ptr=s_e8m0,
        output_ptr=output,
        n_elements_in=n_elements_in,
        SIGN_MASK_F4=0x8,
        EXP_MASK_F4=0x6,
        EXP_SHIFT_F4=1,
        EXP_BIAS_F4=3,
        MANTISSA_MASK_F4=0x1,
        MANTISSA_BITS_F4=1,
        BLOCK_SIZE=mx_block_size
    )
    
    return output.reshape(*x.shape[:-1], -1)  # Maintain original batch dimensions
