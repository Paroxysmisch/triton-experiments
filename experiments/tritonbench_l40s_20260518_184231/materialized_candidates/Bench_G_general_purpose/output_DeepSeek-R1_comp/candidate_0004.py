import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _swiglu_fwd_kernel(
    X, Y, OUT,
    M, N,  # M is number of rows, N is number of columns
    stride_x_row, stride_x_col,
    stride_y_row, stride_y_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    if row_idx >= M:
        return
    
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    x_ptr = X + row_idx * stride_x_row + col_offsets * stride_x_col
    y_ptr = Y + row_idx * stride_y_row + col_offsets * stride_y_col
    out_ptr = OUT + row_idx * stride_out_row + col_offsets * stride_out_col
    
    x = tl.load(x_ptr, mask=mask, other=0.0)
    y = tl.load(y_ptr, mask=mask, other=0.0)
    
    sig_x = tl.sigmoid(x)
    output = x * sig_x * y
    
    tl.store(out_ptr, output, mask=mask)

def _swiglu_fwd(xy: torch.Tensor):
    assert xy.shape[-1] % 2 == 0, "Input tensor must have even last dimension"
    
    # Split input into x and y along the last dimension
    x, y = torch.tensor_split(xy, 2, dim=-1)
    x = x.contiguous()
    y = y.contiguous()
    
    # Reshape to 2D tensors (M, N)
    *dims, N = x.shape
    M = torch.tensor(dims).prod().item() if dims else 1
    x_2d = x.view(M, N)
    y_2d = y.view(M, N)
    
    # Initialize output tensor
    out = torch.empty_like(x_2d)
    
    # Kernel grid configuration
    def grid(meta):
        return (M, triton.cdiv(N, meta['BLOCK_SIZE']))
    
    # Launch kernel
    _swiglu_fwd_kernel[grid](
        x_2d, y_2d, out, M, N,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        out.stride(0), out.stride(1),
    )
    
    # Reshape output to match input shape
    return out.view(x.shape)

# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(10, 512, device='cuda')  # (M, 2*N)
    output = _swiglu_fwd(input_tensor)
    print(output.shape)  # Should be (10, 256)
