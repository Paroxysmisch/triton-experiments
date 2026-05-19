import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _swiglu_fwd_kernel(
    X_ptr,
    Y_ptr,
    OUT_ptr,
    M,  # Number of rows
    N,  # Number of columns
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Check if row index is within bounds
    if row_idx >= M:
        return
    
    # Compute column offsets for this block
    col_start = col_block_idx * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load X and Y values
    x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets * stride_y_col
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    
    # Compute SwiGLU: X * sigmoid(Y)
    sig_y = tl.sigmoid(y)
    output = x * sig_y
    
    # Store the result
    out_ptrs = OUT_ptr + row_idx * stride_out_row + col_offsets * stride_out_col
    tl.store(out_ptrs, output, mask=mask)

def _swiglu_fwd(xy: torch.Tensor) -> torch.Tensor:
    # Split input tensor into x and y along the last dimension
    assert xy.dim() == 2, "Input tensor must be 2-dimensional"
    assert xy.size(-1) % 2 == 0, "Last dimension must be even"
    x = xy[..., :xy.size(-1) // 2].contiguous()
    y = xy[..., xy.size(-1) // 2:].contiguous()
    
    M, N = x.shape
    out = torch.empty_like(x)
    
    # Define grid as a function of kernel parameters
    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_SIZE']))
    
    # Launch kernel
    _swiglu_fwd_kernel[grid](
        x, y, out,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
    )
    return out
