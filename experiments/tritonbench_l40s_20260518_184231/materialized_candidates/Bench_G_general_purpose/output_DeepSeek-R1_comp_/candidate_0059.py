import torch
import triton
import triton.language as tl

@triton.jit
def _fp4_packed_to_bf16(f4_val):
    sign = (f4_val >> 3) & 0x1
    exp = (f4_val >> 1) & 0x3
    mantissa = f4_val & 0x1

    # Special cases: exp == 0
    is_zero = (exp == 0) & (mantissa == 0)
    is_denormal = (exp == 0) & (mantissa != 0)

    # Compute components for normal numbers
    sign_f32 = sign << 31
    exp_f32 = (exp + 126) << 23
    mantissa_f32 = mantissa << 22
    fp32_bits_normal = sign_f32 | exp_f32 | mantissa_f32

    # Apply special cases
    fp32_bits = tl.where(is_zero, 0, fp32_bits_normal)
    fp32_bits = tl.where(is_denormal, 0x3F000000, fp32_bits)

    # Convert to FP32 and then to BF16
    fp32_val = tl.bitcast(fp32_bits, tl.float32)
    bf16_val = tl.math.fp32_to_bf16(fp32_val)
    return bf16_val

@triton.jit
def triton_f4_to_bf16_kernel(
    input_ptr,
    output_ptr,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_numel = output_numel // 2
    mask = offsets < input_numel

    input_bytes = tl.load(input_ptr + offsets, mask=mask, other=0)

    # Extract low and high 4-bit values
    low_f4 = input_bytes & 0x0F
    high_f4 = (input_bytes >> 4) & 0x0F

    # Convert to BF16
    bf16_low = _fp4_packed_to_bf16(low_f4)
    bf16_high = _fp4_packed_to_bf16(high_f4)

    # Calculate output offsets
    output_offsets_low = 2 * offsets
    output_offsets_high = 2 * offsets + 1

    # Store results with mask
    tl.store(output_ptr + output_offsets_low, bf16_low, mask=mask)
    tl.store(output_ptr + output_offsets_high, bf16_high, mask=mask)

def triton_f4_to_bf16(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.dtype == torch.uint8, "Input must be uint8 for packed FP4"
    assert x.is_cuda, "Input must be on CUDA"

    x_flat = x.view(-1)
    output_numel = x_flat.numel() * 2
    output = torch.empty((output_numel,), dtype=torch.bfloat16, device=x.device)

    BLOCK_SIZE = 1024  # Tune based on your GPU's optimal settings
    grid = lambda meta: (triton.cdiv(x_flat.numel(), meta['BLOCK_SIZE']), )

    triton_f4_to_bf16_kernel[grid](
        x_flat, output, output_numel, BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape output to have the same number of dimensions as input but last dimension doubled
    original_shape = x.shape
    new_shape = original_shape[:-1] + (original_shape[-1] * 2,)
    return output.view(new_shape)
