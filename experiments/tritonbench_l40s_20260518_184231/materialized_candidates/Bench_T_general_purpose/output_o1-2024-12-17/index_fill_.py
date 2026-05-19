import torch
import triton
import triton.language as tl

@triton.jit
def _index_fill2d_dim1_kernel(
    ptr_x,             # data pointer to the tensor
    stride0,           # row stride
    stride1,           # column stride
    ptr_index,         # data pointer to the index tensor
    n_index,           # number of indices
    value,             # fill value (float)
    n_rows,            # number of rows in 'x'
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes a contiguous block of rows
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_idx = tl.program_id(0) * BLOCK_SIZE + row_offsets
    in_bounds = row_idx < n_rows

    # Loop over each index in 'index' and fill those columns in-place
    for offset in range(n_index):
        col_idx = tl.load(ptr_index + offset)
        # Compute output pointer for the (row_idx, col_idx) location
        out_ptr = ptr_x + row_idx * stride0 + col_idx * stride1
        tl.store(out_ptr, value, mask=in_bounds)

def index_fill_(x: torch.Tensor, dim: int, index: torch.Tensor, value: float) -> torch.Tensor:
    """
    In-place version of:
    index_fill_(dim, index, value) -> Tensor

    Fills elements of x (2D) along the given dimension with 'value',
    using the provided 'index' to select which indices to fill.
    Modifies x in-place. Currently supports dim = 1 only, 2D tensors.
