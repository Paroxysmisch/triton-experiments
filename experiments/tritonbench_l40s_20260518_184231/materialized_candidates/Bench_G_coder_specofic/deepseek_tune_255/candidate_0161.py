import torch
import triton
import triton.language as tl
from vllm.model_executor.ops.triton.triton_utils import calculate_lastdim_num_blocks

@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    SIGN_MASK_F4: tl.constexpr,
    SIGN_SHIFT_F4: tl.constexpr,
    SIGN_MASK_BF16: tl.constexpr,
    SIGN_SHIFT_BF16: tl.constexpr,
    MANT_MASK_F4: tl.constexpr,
    MANT_MASK_BF16: tl.constexpr,
    MANT_SHIFT_BF16: tl.constexpr,
    N_MANT_BITS_F4: tl.constexpr,
    N_MANT_BITS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_in
    x = tl.load(x_ptr + offsets, mask=mask)
    s = tl.load(s_ptr)
    sign_f4 = x & SIGN_MASK_F4
    mant_f4 = x & MANT_MASK_F4
    # Shift mantissa to align MSB of bf16 mantissa
    mant_f4 = mant_f4 >> (N_MANT_BITS_F4 - N_MANT_BITS_BF16)
    # Correct zero and denormal
    is_zero = mant_f4 == 0
    mant_bf16 = mant_f4
    # Apply scale factor
    mant_bf16 = (mant_bf16 * s) >> (16 + N_MANT_BITS_BF16)
    # Correct sign
    sign_bf16 = sign_f4 >> SIGN_SHIFT_F4
    sign_bf16 = sign_bf16 << SIGN_SHIFT_BF16
    # Final result
    result = sign_bf16 | mant_bf16
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_f4_to_scaled_bf16(
    x: torch.Tensor, s_e8m0: torch.Tensor, mx_block_size: int = 512
) -> torch.Tensor:
    output_shape = x.shape
    output_tensor = torch.empty(
        output_shape, dtype=torch.bfloat16, device=x.device
    )
    n_elements_out = x.numel()
    n_blocks = calculate_lastdim_num_blocks(x, mx_block_size)
    x = x.contiguous()
    assert x.is_cuda
    triton_f4_to_scaled_bf16_kernel[n_blocks,](
        x,
        s_e8m0,
        output_tensor,
        n_elements_out,
        SIGN_MASK_F4=0b1000,
        SIGN_SHIFT_F4=3,
        SIGN_MASK_BF16=0b10000000,
        SIGN_SHIFT_BF16=7,
        MANT_MASK_F4=0b1111,
        MANT_MASK_BF16=0b11111111,
        MANT_SHIFT_BF16=0,
        N_MANT_BITS_F4=4,
        N_MANT_BITS_BF16=16,
        BLOCK_SIZE=512,
    )
    return output_tensor
