import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
    ],
    key=['n_elements']
)
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    absmax_inv = tl.load(absmax_inv_ptr)

    x = tl.load(x_ptr + offsets, mask=mask)
    scaled = x * absmax_inv
    quantized = tl.extra.cuda.libdevice.llrint(scaled)
    quantized_int8 = quantized.to(tl.int8)
    tl.store(output_ptr + offsets, quantized_int8, mask=mask)

def quantize_global(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    absmax = torch.max(torch.abs(x))
    absmax_inv = torch.tensor(1.0 / absmax, dtype=torch.float32, device=x.device)

    output = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    _quantize_global[grid](
        x,
        absmax_inv,
        output,
        n_elements,
        BLOCK_SIZE=1024,  # Initial value, autotune will override
    )

    return output, absmax
