import triton
import triton.language as tl

@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input block
    mask = offsets < n_elements
    x_block = tl.load(x_ptr + offsets, mask=mask)

    # Load the inverse of the maximum absolute value
    absmax_inv = tl.load(absmax_inv_ptr)

    # Apply the quantization formula
    quantized_block = tl.round(x_block * absmax_inv).to(tl.int8)

    # Store the quantized block
    tl.store(output_ptr + offsets, quantized_block, mask=mask)

import torch
import triton
import triton.runtime

def quantize_global(x: torch.Tensor):
    # Ensure the input tensor is on the GPU
    assert x.is_cuda, "Input tensor must be on the GPU"

    # Compute the maximum absolute value
    absmax = x.abs().max()
    absmax_inv = 127.0 / absmax

    # Allocate memory for the output tensor
    output = torch.empty_like(x, dtype=torch.int8, device=x.device)

    # Prepare the pointers
    x_ptr = x.data_ptr()
    absmax_inv_ptr = absmax_inv.data_ptr()
    output_ptr = output.data_ptr()

    # Define the grid size
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    _quantize_global[grid_size, BLOCK_SIZE](x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE)

    return output, absmax
