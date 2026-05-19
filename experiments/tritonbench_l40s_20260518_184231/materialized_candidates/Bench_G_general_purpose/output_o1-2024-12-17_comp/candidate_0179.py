import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _quantize_global(
    x_ptr,            # *f32
    absmax_inv_ptr,   # *f32
    output_ptr,       # *i8
    n_elements,       # i32
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_vals = tl.where(mask, tl.load(x_ptr + offsets), 0.0)
    absmax_inv = tl.load(absmax_inv_ptr)
    
    scaled = x_vals * absmax_inv
    quant = tl.libdevice.llrint(scaled)
    quant_clamped = tl.max(tl.min(quant, 127), -128)
    quant_i8 = quant_clamped.to(tl.int8)

    tl.store(output_ptr + offsets, quant_i8, mask=mask)

def quantize_global(x: torch.Tensor):
    absmax = x.abs().max()
    # Avoid division by zero if absmax is 0
    absmax_inv = 1.0 / absmax if absmax != 0 else 0.0

    # Create output tensor of int8 type
    out = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()

    # Launch the kernel
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _quantize_global[grid](x, absmax_inv, out, n_elements)

    return out, absmax.item()
