import torch
import triton
import triton.language as tl

# Constants
SIGN_MASK_F4 = 0x8
ZERO_BITS_F32 = 0x00000000
INF_BITS_F32 = 0x7F800000
EXPONENT_BIAS_F4 = 7
EXPONENT_BIAS_F32 = 127
EXPONENT_BIAS_DIFF = EXPONENT_BIAS_F32 - EXPONENT_BIAS_F4
EXPONENT_MASK_F4 = 0x7
MANTISSA_MASK_F4 = 0x0
BLOCK_SIZE_IN = 1024

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr, s_ptr, output_ptr, n_elements_in,
    SIGN_MASK_F4, ZERO_BITS_F32, INF_BITS_F32,
    EXPONENT_BIAS_DIFF, EXPONENT_MASK_F4, MANTISSA_MASK_F4,
    BLOCK_SIZE_IN: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_IN
    offsets = block_start + tl.arange(0, BLOCK_SIZE_IN)
    mask = offsets < n_elements_in

    # Load packed f4 values
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Unpack f4 values
    sign = (x & SIGN_MASK_F4) << 24
    exponent = ((x & EXPONENT_MASK_F4) + EXPONENT_BIAS_DIFF) << 23
    mantissa = (x & MANTISSA_MASK_F4) << 19

    # Combine components
    f32_bits = sign | exponent | mantissa

    # Handle special cases
    is_zero = x == 0
    is_inf = (x & 0x7) == 0x7
    f32_bits = tl.where(is_zero, ZERO_BITS_F32, f32_bits)
    f32_bits = tl.where(is_inf, INF_BITS_F32, f32_bits)

    # Convert to float32
    f32_val = tl.bitcast(f32_bits, tl.float32)

    # Apply scaling
    scale = tl.load(s_ptr)
    scaled_val = f32_val * scale

    # Convert to bf16
    bf16_val = tl.float32_to_bfloat16(scaled_val)

    # Store result
    tl.store(output_ptr + offsets, bf16_val, mask=mask)

def triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size=BLOCK_SIZE_IN):
    # Input checks
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert s_e8m0.numel() == 1, "Scale must be a scalar"
    
    # Prepare output tensor
    output = torch.empty(x.numel(), dtype=torch.bfloat16, device=x.device)
    
    # Calculate grid size
    grid = (x.numel() + mx_block_size - 1) // mx_block_size
    
    # Launch kernel
    triton_f4_to_scaled_bf16_kernel[(grid,)](
        x, s_e8m0, output, x.numel(),
        SIGN_MASK_F4, ZERO_BITS_F32, INF_BITS_F32,
        EXPONENT_BIAS_DIFF, EXPONENT_MASK_F4, MANTISSA_MASK_F4,
        BLOCK_SIZE_IN=mx_block_size
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample input
    x = torch.randint(0, 16, (1000,), dtype=torch.uint8, device='cuda')
    s_e8m0 = torch.tensor([2.0], dtype=torch.float32, device='cuda')
    
    # Run conversion
    result = triton_f4_to_scaled_bf16(x, s_e8m0)
    print(result.shape, result.dtype)
