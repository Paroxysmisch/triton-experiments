import triton
import triton.language as tl

# Constants
SIGN_MASK_F4 = 0x80
MANTISSA_MASK_F4 = 0x7F
ZERO_BITS_F32 = 0x00000000
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000
F4_E2M1_EXP_BIAS = 7
F32_EXP_BIAS = 127

# Helper function to convert FP4 to BF16
@triton.jit
def _fp4_packed_to_bf16(x: tl.int8) -> tl.float16:
    sign = (x & SIGN_MASK_F4) >> 7
    mantissa = x & MANTISSA_MASK_F4
    exponent = (mantissa >> 3) - F4_E2M1_EXP_BIAS + F32_EXP_BIAS

    if mantissa == 0:
        return tl.float16(0.0) if sign == 0 else tl.float16(-0.0)
    elif mantissa == 0x40:
        return tl.float16(0.5) if sign == 0 else tl.float16(-0.5)
    else:
        mantissa = (mantissa & 0x07) << 10
        return tl.float16(tl.make_int32((sign << 15) | (exponent << 10) | mantissa))

# Triton kernel for converting FP4 to BF16
@triton.jit
def triton_f4_to_bf16_kernel(
    x_ptr: tl.pointer_type(tl.int8, 1),
    output_ptr: tl.pointer_type(tl.float16, 1),
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_block = tl.load(x_ptr + offsets, mask=mask)
    output_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float16)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            output_block[i] = _fp4_packed_to_bf16(x_block[i])

    tl.store(output_ptr + offsets, output_block, mask=mask)

import torch
import triton
import triton.language as tl

def triton_f4_to_bf16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8, "Input tensor must be of type int8"
    assert x.is_cuda, "Input tensor must be on a CUDA device"

    n_elements = x.numel()
    output = torch.empty(n_elements, dtype=torch.bfloat16, device=x.device)

    # Compute grid size
    BLOCK_SIZE = 128
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    triton_f4_to_bf16_kernel[grid_size, BLOCK_SIZE](
        x, output, n_elements, BLOCK_SIZE
    )

    return output
