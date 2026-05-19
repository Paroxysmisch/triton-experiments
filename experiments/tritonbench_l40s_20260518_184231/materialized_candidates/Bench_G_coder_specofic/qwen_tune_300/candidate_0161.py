import torch
import triton
import triton.language as tl
from .triton_utils import get_kernel_meta

# Triton kernel for converting packed fp4 to scaled bf16
@triton.jit
def triton_f4_to_scaled_bf16_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    n_elements_in,
    # Meta-params
    BLOCK_SIZE: tl.constexpr,
    # Constants
    SIGN_MASK_F4: tl.constexpr,
    MANTISSA_MASK_F4: tl.constexpr,
    MANTISSA_BF16: tl.constexpr,
    BITS_PER_ELEMENT: tl.constexpr,
):
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_in
    # Load elements and scaling factors
    x = tl.load(x_ptr + (offsets // (BITS_PER_ELEMENT // 8)), mask=mask)
    s = tl.load(s_ptr + offsets, mask=mask)
    # Convert and scale
    output = tl.f4_to_bf16(x, SIGN_MASK_F4, MANTISSA_MASK_F4, MANTISSA_BF16) * s
    # Handle special cases
    output = tl.where(
        mask,
        output,
        0,
    )
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Function to invoke the Triton kernel
def triton_f4_to_scaled_bf16(x, s_e8m0, mx_block_size=None):
    # Determine output shape and size
    output_shape = list(x.shape[:-1]) + [x.shape[-1] * 2]
    output = torch.empty(output_shape, device=x.device, dtype=torch.bfloat16)
    n_elements_out = output.numel()
    n_elements_in = x.numel()
    # Configure kernel grid
    grid = lambda meta: (triton.cdiv(n_elements_out, meta["BLOCK_SIZE"]),)
    # Launch kernel
    triton_f4_to_scaled_bf16_kernel[grid](
        x,
        s_e8m0,
        output,
        n_elements_in,
        BLOCK_SIZE=mx_block_size,
        **get_kernel_meta(x),
    )
    return output
