import torch
import triton
import triton.language as tl
import math

# Triton kernel for global quantization
@triton.autotune(configs=triton.Configs({"BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096]}, num_stages=3, num_warps=8), key=["n_elements"])
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the block index and the element offsets within the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load the elements and the precomputed inverse of the max abs value
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    absmax_inv = tl.load(absmax_inv_ptr)

    output = tl.libdevice.llrint(127.0 * (x * absmax_inv))
    # Store the result
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)

def quantize_global(x: torch.Tensor):
    # Calculate abs max
    absmax = x.abs().max().unsqueeze(0)
    # Calculate absmax_inv
    absmax_inv = absmax.clone().to(torch.get_default_dtype())
    absmax_inv = 1.0 / absmax_inv
    # Prepare output
    output = torch.empty_like(x, dtype=torch.int8, device=x.device)

    # Ensure CUDA use
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    # Define grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the Triton kernel
    _quantize_global[grid](x, absmax_inv, output, n_elements, BLOCK_SIZE=1024)

    return output, absmax
