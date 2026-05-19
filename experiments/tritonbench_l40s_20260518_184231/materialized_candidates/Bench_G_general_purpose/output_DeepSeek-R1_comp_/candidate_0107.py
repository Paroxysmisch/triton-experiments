import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(
    # Pointers to tensors
    x_ptr, y_ptr, do_ptr,
    dx_ptr, dy_ptr, out_ptr,
    # Strides for rows in each tensor
    x_row_stride, y_row_stride, do_row_stride,
    dx_row_stride, dy_row_stride, out_row_stride,
    n_cols,  # Number of columns per row
    BLOCK_N: tl.constexpr,  # Number of columns processed per block
    RECOMPUTE_OUTPUT: tl.constexpr,  # Whether to recompute output
):
    # 2D grid: row_idx and block_idx determine current block
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    col_offset = block_idx * BLOCK_N
    
    # Generate column indices and mask for valid columns
    col_indices = col_offset + tl.arange(0, BLOCK_N)
    mask = col_indices < n_cols

    # Compute pointers for the current row and block
    x_row = x_ptr + row_idx * x_row_stride
    y_row = y_ptr + row_idx * y_row_stride
    do_row = do_ptr + row_idx * do_row_stride
    dx_row = dx_ptr + row_idx * dx_row_stride
    dy_row = dy_ptr + row_idx * dy_row_stride

    # Load input data
    x = tl.load(x_row + col_indices, mask=mask)
    y = tl.load(y_row + col_indices, mask=mask)
    do = tl.load(do_row + col_indices, mask=mask)

    # Compute sigmoid and its derivative
    s = tl.sigmoid(y)
    ds = s * (1 - s)

    # Calculate gradients
    dx = do * s
    dy = do * x * ds

    # Store gradients
    tl.store(dx_row + col_indices, dx, mask=mask)
    tl.store(dy_row + col_indices, dy, mask=mask)

    # Recompute and store output if needed
    if RECOMPUTE_OUTPUT:
        out = x * s
        out_row = out_ptr + row_idx * out_row_stride
        tl.store(out_row + col_indices, out, mask=mask)

def _swiglu_bwd(do: torch.Tensor, xy: torch.Tensor, out: torch.Tensor = None, recompute_output: bool = False):
    # Ensure contiguous tensors
    do = do.contiguous()
    xy = xy.contiguous()

    # Reshape to 2D (batch*seqlen, hid_size)
    batch_dims = xy.shape[:-1]
    num_rows = 1
    for d in batch_dims:
        num_rows *= d
    n_features = xy.shape[-1]
    assert n_features % 2 == 0, "Input tensor must have even last dimension"
    n_cols = n_features // 2

    xy_2d = xy.view(num_rows, -1)
    x = xy_2d[:, :n_cols]
    y = xy_2d[:, n_cols:]

    # Initialize gradient tensors
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)

    # Handle output tensor
    if recompute_output:
        if out is None:
            out = torch.empty_like(do)
        else:
            assert out.shape == do.shape, "Output tensor shape mismatch"
    else:
        out = torch.tensor([])  # Dummy tensor if not needed

    # Kernel configuration
    BLOCK_N = 128  # Tunable based on hardware
    num_blocks_per_row = (n_cols + BLOCK_N - 1) // BLOCK_N
    grid = (num_rows, num_blocks_per_row)

    # Launch kernel
    _swiglu_bwd_kernel[grid](
        x, y, do, dx, dy, out,
        x.stride(0), y.stride(0), do.stride(0),
        dx.stride(0), dy.stride(0), out.stride(0),
        n_cols,
        BLOCK_N=BLOCK_N,
        RECOMPUTE_OUTPUT=recompute_output,
    )

    # Reshape outputs to match input batch dimensions
    dx = dx.view(*batch_dims, n_cols)
    dy = dy.view(*batch_dims, n_cols)
    if recompute_output:
        out = out.view(*batch_dims, n_cols)
        return dx, dy, out
    return dx, dy
