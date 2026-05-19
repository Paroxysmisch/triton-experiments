import triton
import triton.language as tl
import torch

# Constants for FP4 and FP32 formats
SIGN_MASK_F4 = 0x8
MANTISSA_MASK_F4 = 0x1
ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3f000000
EBITS_F4_E2M1 = 2
MBITS_F4_E2M1 = 1
EBITS_F32 = 8
MBITS_F32 = 23
BIAS_F4_E2M1 = 1
BIAS_F32 = 127

@triton.jit
def _fp4_packed_to_bf16(packed_val):
    # Extract low and high nibbles
    low_nibble = packed_val & 0xF
    high_nibble = (packed_val >> 4) & 0xF
    
    # Process both nibbles
    def process_nibble(nibble):
        # Extract sign, exponent, and mantissa
        sign = (nibble & SIGN_MASK_F4) << (31 - 3)  # Move to FP32 sign position
        exp = (nibble >> 1) & 0x3  # 2-bit exponent
        mantissa = nibble & MANTISSA_MASK_F4
        
        # Handle special cases
        is_zero = exp == 0 and mantissa == 0
        is_denormal = exp == 0 and mantissa != 0
        
        # Adjust exponent
        if not is_zero and not is_denormal:
            # Normal number: adjust exponent bias
            exp = (exp + BIAS_F32 - BIAS_F4_E2M1) << (EBITS_F32 - EBITS_F4_E2M1)
        else:
            exp = 0
        
        # Handle denormal case (0.5)
        if is_denormal:
            return ZERO_POINT_FIVE_BITS_F32 | sign
        
        # Handle zero case
        if is_zero:
            return ZERO_BITS_F32
        
        # Shift mantissa to proper position
        mantissa = mantissa << (MBITS_F32 - MBITS_F4_E2M1)
        
        # Combine components
        return sign | exp | mantissa
    
    # Convert both nibbles to BF16
    val1 = process_nibble(low_nibble)
    val2 = process_nibble(high_nibble)
    
    return tl.float32_to_bfloat16(val1), tl.float32_to_bfloat16(val2)

@triton.jit
def triton_f4_to_bf16_kernel(
    input_ptr,  # Pointer to packed FP4 input
    output_ptr,  # Pointer to BF16 output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Load packed values
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements // 2  # Divide by 2 because each byte contains 2 FP4 values
    
    packed_values = tl.load(input_ptr + offsets, mask=mask)
    
    # Process each packed value
    for i in range(BLOCK_SIZE):
        if i < n_elements // 2:
            val1, val2 = _fp4_packed_to_bf16(packed_values[i])
            
            # Store results
            output_idx = block_start * 2 + i * 2
            if output_idx < n_elements:
                tl.store(output_ptr + output_idx, val1)
            if output_idx + 1 < n_elements:
                tl.store(output_ptr + output_idx + 1, val2)

def triton_f4_to_bf16(input_tensor):
    assert input_tensor.is_cuda and input_tensor.is_contiguous()
    n_elements = input_tensor.numel() * 2  # Each byte contains 2 FP4 values
    
    # Allocate output tensor
    output = torch.empty(n_elements, dtype=torch.bfloat16, device=input_tensor.device)
    
    # Calculate grid dimensions
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * 2),)  # Divide by 2 for packed values
    
    # Launch kernel
    triton_f4_to_bf16_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
