import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements, BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * P2
    row_end = row_start + P2

    # Load the row into a Triton block
    row = tl.load(x_ptr + row_start, mask=row_start + tl.arange(0, P2) < n_elements, other=0.0)

    # Compute the absolute values of the row elements
    abs_row = tl.abs(row)

    # Find the maximum value in the row
    max_val = tl.max(abs_row, axis=0)

    # Store the max value in the output_maxs array
    tl.store(output_maxs + pid, max_val)

    # Quantize the row elements
    quantized_row = tl.extra.cuda.libdevice.llrint(row / max_val * 127.0)

    # Store the quantized row in the output tensor
    tl.store(output_ptr + row_start, quantized_row, mask=row_start + tl.arange(0, P2) < n_elements)

import torch
import triton
import triton.language as tl

def quantize_rowwise(x):
    assert x.is_cuda, "Input tensor must be a CUDA tensor"
    assert x.dim() == 2, "Input tensor must be 2D"

    n_rows, n_cols = x.shape
    n_elements = n_rows * n_cols

    # Allocate output tensors
    output = torch.empty((n_rows, n_cols), dtype=torch.int8, device=x.device)
    output_maxs = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    # Define block size and power of 2 ceiling of the row size
    BLOCK_SIZE = 128
    P2 = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    # Launch the kernel
    grid = (n_rows,)
    _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE, P2)

    return output, output_maxs
