import triton
import triton.language as tl
import torch

SIGN_MASK_F4 = 0x8
EXP_MASK_F4 = 0x7
BIAS_DIFFERENCE = (127 - 7) << 7  # BF16 bias (127) - example f4 bias (7), shift exponent bits to proper position
MANTISSA_SHIFT = 6               # Shift for moving 1-bit mantissa to BF16 lower bits
SIGN_SHIFT = 15                  # Sign bit position for BF16

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,      # *uint8
    s_ptr,      # *uint16 (e8m0 scaling exponent bits)
    output_ptr, # *uint16 (bf16)
    n_elements_in, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements_in

    # Load the packed 4-bit values; each element i is stored in x_ptr[i // 2]
    byte_vals = tl.load(x_ptr + (offs // 2), mask=mask, other=0)
    # Decide whether to pick high/low nibble
    nibble_is_high = (offs % 2) == 0
    nibble_shift = tl.where(nibble_is_high, 4, 0)
    nibble = (byte_vals >> nibble_shift) & 0xF

    # Decode sign, exponent, mantissa from nibble
    sign_bit = (nibble & SIGN_MASK_F4) >> 3
    exp_bits = nibble & EXP_MASK_F4
    mant_bit = nibble & 0x1

    # Special case: if nibble == 0, result is zero
    is_zero = nibble == 0

    # Form bf16 from sign, exponent, mantissa
    # move exponent into BF16 position, add bias difference
    bf16_exp = ((exp_bits << 7) + BIAS_DIFFERENCE)
    bf16_sign = sign_bit << SIGN_SHIFT
    bf16_mant = mant_bit << MANTISSA_SHIFT
    bf16_val = bf16_sign + bf16_exp + bf16_mant

    # Zero out if nibble == 0
    bf16_val = tl.where(is_zero, 0, bf16_val)

    # Load e8m0 scaling factor (16 bits, interpret exponent as offset)
    raw_s = tl.load(s_ptr + offs, mask=mask, other=0)
    # Convert e8m0 => exponent shift
    # raw_s is stored in 16 bits, but we only need lower bits as exponent
    exponent_shift = raw_s & 0xFF
    # Shift sign if exists in scaling factor (if any)
    sign_s = tl.broadcast_to(0, exponent_shift.shape)  # no sign bit usage here
    sign_s_shift = sign_s << SIGN_SHIFT

    # Combine sign and exponent shift
    combined_s = sign_s_shift + ((exponent_shift + 127) << 7)  # Shift exponent relative to BF16 bias

    # Multiply bf16_val by 2^(exponent_shift) in BF16 domain:
    # We can just add exponents in BF16
    # (bf16_val exponent) + exponent_shift => new exponent in BF16
    # Extract sign, exp, mant from bf16_val
    bf16_sign_part = (bf16_val >> SIGN_SHIFT) & 0x1
    bf16_exp_part = (bf16_val >> 7) & 0xFF
    bf16_mant_part = bf16_val & 0x7F

    # Add exponent
    scaled_exp_part = bf16_exp_part + exponent_shift
    # If scaled exponent goes beyond 255, clamp to avoid inf
    scaled_exp_part = tl.max(tl.min(scaled_exp_part, 255), 0)

    # Reconstruct scaled BF16
    scaled_bf16_val = (bf16_sign_part << SIGN_SHIFT) + (scaled_exp_part << 7) + bf16_mant_part

    # Store result
    tl.store(output_ptr + offs, scaled_bf16_val, mask=mask)


def triton_f4_to_scaled_bf16(x: torch.Tensor, s_e8m0: torch.Tensor, mx_block_size=256) -> torch.Tensor:
    assert x.dtype == torch.uint8, "Input must be packed 4-bit in uint8 format"
    assert s_e8m0.dtype == torch.uint16, "Scaling factors must be stored in uint16"
    x_contig = x.contiguous()
    s_e8m0_contig = s_e8m0.contiguous()

    # Number of final bf16 elements is 2 * x.numel(), each nibble is one f4
    n_elements_in = x_contig.numel() * 2
    assert s_e8m0_contig.numel() == n_elements_in, "Scaling tensor size must match number of f4 elements"

    # Prepare output tensor
    out = torch.empty(n_elements_in, dtype=torch.uint16, device=x.device)

    grid = lambda META: ( (n_elements_in + mx_block_size - 1) // mx_block_size, )
    triton_f4_to_scaled_bf16_kernel[grid](
        x_contig, 
        s_e8m0_contig, 
        out, 
        n_elements_in, 
        BLOCK_SIZE=mx_block_size
    )
    return out
