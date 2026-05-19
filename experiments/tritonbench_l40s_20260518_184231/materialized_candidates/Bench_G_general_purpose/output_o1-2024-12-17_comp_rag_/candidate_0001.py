import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=8),
    ],
    key=["ncols"],
)
@triton.jit
def _swiglu_fwd_kernel(
    X_ptr, 
    Y_ptr, 
    OUT_ptr, 
    nrows,
    ncols,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_start = col_block_idx * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    col_idx = col_start + cols

    # Check bounds
    mask = col_idx < ncols

    # Compute pointers
    x_offset = row_idx * stride_x_row + col_idx * stride_x_col
    y_offset = row_idx * stride_y_row + col_idx * stride_y_col
    out_offset = row_idx * stride_out_row + col_idx * stride_out_col

    # Load X and Y
    x_data = tl.load(X_ptr + x_offset, mask=mask, other=0.0)
    y_data = tl.load(Y_ptr + y_offset, mask=mask, other=0.0)

    # Swiglu = x_data * sigmoid(x_data) * y_data
    # Sigmoid is 1 / (1 + exp(-x_data))
    sig = 1.0 / (1.0 + tl.exp(-x_data))
    swiglu_val = x_data * sig * y_data

    # Store result
    tl.store(OUT_ptr + out_offset, swiglu_val, mask=mask)


def _swiglu_fwd(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy.chunk(2, dim=-1)
    x = x.contiguous()
    y = y.contiguous()

    M, N = x.shape
    out = torch.empty_like(x)

    grid = (M, triton.cdiv(N, 64))
    _swiglu_fwd_kernel[grid](
        x, 
        y, 
        out, 
        M, 
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out
