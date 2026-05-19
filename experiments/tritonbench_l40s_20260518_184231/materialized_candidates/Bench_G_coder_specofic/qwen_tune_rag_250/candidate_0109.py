import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    TILE_N: tl.constexpr,
):
    # The softmax is applied row-wise, so we store the row index in the inner loop
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The tile size is the next power of two greater than n_cols, so we can fit each
    # row in a single tile
    tile_n_offsets = tl.arange(0, TILE_N)
    input_ptrs = row_start_ptr + tile_n_offsets
    # Load the row into SRAM, using a mask since TILE_N may be > than n_cols
    tile_n = tl.load(input_ptrs, mask=tile_n_offsets < n_cols, other=-float("inf"))
    # Subtract maximum for numerical stability
    row_minus_max = tile_n - tl.max(tile_n, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + tile_n_offsets
    tl.store(output_ptrs, softmax_output, mask=tile_n_offsets < n_cols)

@torch.no_grad()
def softmax(x, dim=-1):
    # Only need to support 2D tensor as of now
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    # The tile size is the smallest power of two greater than the number of columns in `x`
    TILE_N = triton.next_power_of_2(n_cols)
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix
    softmax_kernel_online_v2[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        TILE_N=TILE_N,
    )
    return y
