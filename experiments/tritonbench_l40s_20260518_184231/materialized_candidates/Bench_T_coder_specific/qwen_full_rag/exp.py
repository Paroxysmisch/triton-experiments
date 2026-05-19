import triton
import triton.language as tl
import torch

@triton.jit
def exp_kernel(x_ptr, x_shape, x_stride, y_ptr, y_shape, y_stride, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(x_shape[1], BLOCK_N)
    row_offset = pid // grid_n
    col_offset = pid % grid_n
    x_row_offset = row_offset * x_stride
    y_row_offset = row_offset * y_stride
    for k in range(0, x_shape[2]):
        x_col_offset = col_offset * BLOCK_N
        y_col_offset = col_offset * BLOCK_N
        for i in range(0, BLOCK_N):
            x_offset = x_row_offset + (x_col_offset + i) * x_stride
            y_offset = y_row_offset + (y_col_offset + i) * y_stride
            if x_col_offset + i < x_shape[1]:
                x_val = tl.load(x_ptr + x_offset)
                y_val = tl.exp(x_val)
                tl.store(y_ptr + y_offset, y_val)

def exp(x, out=None):
    shape = x.shape
    ndim = len(shape)
    if ndim > 3:
        raise ValueError("only accept 1D, 2D, 3D inputs")
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
    if ndim == 1:
        N = shape[0]
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), )
        exp_kernel[grid](x, shape, 1, out, shape, 1, BLOCK_N=1024)
    elif ndim == 2:
        M, N = shape
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                             triton.cdiv(N, meta["BLOCK_N"]), )
        exp_kernel[grid](x, shape, 1, out, shape, 1, BLOCK_M=32, BLOCK_N=32)
    else:
        M, N, K = shape
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                             triton.cdiv(N, meta["BLOCK_N"]), )
        exp_kernel[grid](x, shape, 2, out, shape, 2, BLOCK_M=32, BLOCK_N=32)
    return out
