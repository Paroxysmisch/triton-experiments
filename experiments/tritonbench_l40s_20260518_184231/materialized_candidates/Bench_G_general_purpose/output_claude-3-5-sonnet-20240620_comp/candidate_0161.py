import torch
import triton
import triton.language as tl

# Constants for conversion
SIGN_MASK_F4 = 0x8
MASK_F4 = 0x7
BIAS_F4 = 3
BIAS_BF16 = 127
MANTISSA_BITS_F4 = 3
MANTISSA_BITS_BF16 = 7
EXPONENT_BITS_BF16 = 8

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr, s_ptr, output_ptr, n_elements_in,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_in
    
    # Load packed f4 values (2 values per byte)
    x = tl.load(x_ptr + offsets // 2, mask=mask)
    
    # Extract lower and upper 4 bits
    x_low = x & 0xF
    x_high = (x >> 4) & 0xF
    
    # Process lower 4 bits
    sign_low = (x_low & SIGN_MASK_F4) << (15 - 3)
    exp_mantissa_low = x_low & MASK_F4
    bf16_low = tl.where(exp_mantissa_low != 0,
                        sign_low | ((exp_mantissa_low + (BIAS_BF16 - BIAS_F4)) << MANTISSA_BITS_BF16),
                        sign_low)
    
    # Process upper 4 bits
    sign_high = (x_high & SIGN_MASK_F4) << (15 - 3)
    exp_mantissa_high = x_high & MASK_F4
    bf16_high = tl.where(exp_mantissa_high != 0,
                         sign_high | ((exp_mantissa_high + (BIAS_BF16 - BIAS_F4)) << MANTISSA_BITS_BF16),
                         sign_high)
    
    # Combine low and high results
    result = tl.where(offsets % 2 == 0, bf16_low, bf16_high)
    
    # Apply scaling factor
    scale = tl.load(s_ptr + offsets // 8)  # Assuming one scale per 8 elements
    scaled_result = result + ((scale - 127) << 7)  # e8m0 to bf16 exponent adjustment
    
    # Store the result
    tl.store(output_ptr + offsets, scaled_result, mask=mask)

def triton_f4_to_scaled_bf16(x, s_e8m0, BLOCK_SIZE: int = 1024):
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be a CUDA contiguous tensor"
    assert s_e8m0.is_cuda and s_e8m0.is_contiguous(), "Scale tensor must be a CUDA contiguous tensor"
    
    n_elements = x.numel() * 2  # Each byte contains 2 fp4 values
    output = torch.empty(n_elements, dtype=torch.bfloat16, device=x.device)
    
    def grid(meta):
        return (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    triton_f4_to_scaled_bf16_kernel[grid](
        x, s_e8m0, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample input data
    x = torch.randint(0, 256, (1000,), dtype=torch.uint8, device='cuda')
    s_e8m0 = torch.randint(0, 256, (125,), dtype=torch.uint8, device='cuda')  # One scale per 8 elements
    
    # Run the conversion
    result = triton_f4_to_scaled_bf16(x, s_e8m0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
