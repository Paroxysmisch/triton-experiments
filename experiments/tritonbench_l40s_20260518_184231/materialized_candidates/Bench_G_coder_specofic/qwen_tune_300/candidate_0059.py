import torch
import triton
import triton.language as tl

@triton.jit
def _fp4_packed_to_bf16(low_bits, high_bits):
    # Unpack high and low bits
    high_sign = (high_bits >> 7) & 0x01
    high_exponent = ((high_bits >> 3) & 0xF) - 7
    high_mantissa = ((high_bits >> 2) & 0x03) | 0x04
    low_mantissa = (low_bits >> 6) & 0x03

    # Handle special cases
    if (high_exponent == -7) & (high_mantissa == 4):
        # Zero
        high_mantissa = 0
        high_special = 0
    else:
        high_special = 1

    # Convert exponent
    high_exponent = (high_exponent + 12) & 0xFF

    # Reconstruct FP32 value
    x_fp32 = (
        (high_sign << 31)
        | (high_exponent << 23)
        | (high_mantissa << 20)
        | (high_special << 19)
        | (low_mantissa << 17)
    )
    return x_fp32.to(tl.bfloat16)

@triton.jit
def triton_f4_to_bf16_kernel(
    fp4_data_ptr: torch.Tensor,
    output_bf16_tensor: torch.Tensor,
    **meta,
):
    BLOCK_SIZE = meta["BLOCK_SIZE"]
    pid = tl.program_id(axis=0)
    fp4_data_ptr += pid * BLOCK_SIZE
    output_bf16_tensor += pid * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    fp4_data = tl.load(fp4_data_ptr + cols, mask=cols < fp4_data_ptr.shape[0], other=0)
    low_bits = (fp4_data & 0xF0) >> 4
    high_bits = fp4_data & 0x0F
    output = _fp4_packed_to_bf16(low_bits, high_bits)
    tl.store(output_bf16_tensor + cols, output, mask=cols < output_bf16_tensor.shape[0])

def triton_f4_to_bf16(fp4_packed_tensor: torch.Tensor):
    assert fp4_packed_tensor.dtype == torch.int8
    assert fp4_packed_tensor.is_cuda and fp4_packed_tensor.is_contiguous()

    output_bf16_tensor = torch.empty(
        (fp4_packed_tensor.shape[0],), dtype=torch.bfloat16, device="cuda"
    )

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(fp4_packed_tensor.numel(), BLOCK_SIZE),)
    triton_f4_to_bf16_kernel[grid](fp4_packed_tensor, output_bf16_tensor, BLOCK_SIZE=BLOCK_SIZE)

    return output_bf16_tensor
