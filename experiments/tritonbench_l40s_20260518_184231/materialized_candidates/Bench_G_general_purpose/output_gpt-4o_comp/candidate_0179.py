import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
    ],
    key=['n_elements']
)
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    absmax_inv = tl.load(absmax_inv_ptr)

    # Perform quantization
    quantized = tl.libdevice.llrint(x * absmax_inv)

    # Store the result as 8-bit integers
    tl.store(output_ptr + offsets, quantized.to(tl.int8), mask=mask)

import torch

def quantize_global(x):
    # Calculate the maximum absolute value and its reciprocal
    absmax = torch.max(torch.abs(x))
    absmax_inv = 1.0 / absmax if absmax != 0 else 0.0

    # Prepare the output tensor
    output = torch.empty_like(x, dtype=torch.int8)

    # Launch the Triton kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _quantize_global[grid](
        x_ptr=x,
        absmax_inv_ptr=torch.tensor([absmax_inv], device=x.device),
        output_ptr=output,
        n_elements=n_elements
    )

    return output, absmax
