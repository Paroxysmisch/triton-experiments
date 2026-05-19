import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 4096}, num_stages=3, num_warps=8),
    ],
    key=['ncols'],
)
@triton.jit
def _swiglu_fwd_kernel(
    X,
    Y,
    OUT,
    M,
    N,
    ncols,
    stride_x_row,
    stride_y_row,
    stride_out_row,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    block_start = tl.program_id(axis=1) * BLOCK_SIZE_N
    cols = block_start + tl.arange(0, BLOCK_SIZE_N)
    x_mask = cols < ncols
    x = tl.load(X + row * stride_x_row + cols, mask=x_mask)
    y = tl.load(Y + row * stride_y_row + cols, mask=x_mask)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + row * stride_out_row + cols, out, mask=x_mask)

def _swiglu_fwd(xy, y):
    xy = xy.contiguous()
    y = y.contiguous()
    x, xy_b = xy.chunk(2, dim=1)
    B, L = x.shape
    x = x.reshape(B * L)
    y = y.reshape(B * L)
    out = torch.empty_like(x)
    grid = lambda META: (B, triton.cdiv(L, META['BLOCK_SIZE_N']))
    BLOCK_SIZE_M, BLOCK_SIZE_N = (16, 2048)
    with torch.cuda.device(x.device.index):
        _swiglu_fwd_kernel[grid](x, y, out, B, L, ncols=L, stride_x_row=1, stride_y_row=1, stride_out_row=1, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    out = out.reshape(B, L)
    return out
