import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M,
    Out,
    stride_m_x,
    stride_m_y,
    stride_out_x,
    stride_out_y,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    matrix_stridex = tl.arange(0, SIZE_M) * stride_m_x
    matrix_stridey = tl.arange(0, D_HEAD) * stride_m_y
    out_stridex = tl.arange(0, SIZE_M) * stride_out_x
    out_stridey = tl.arange(0, D_HEAD) * stride_out_y

    matrix_ptr = M + matrix_stridex[:, None] + matrix_stridey[None, :]
    out_ptr = Out + out_stridex[:, None] + out_stridey[None, :]
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, matrix)

def wrapper():
    SIZE_M = 16
    D_HEAD = 64

    matrix = torch.rand((SIZE_M, D_HEAD), device="cuda", dtype=torch.float16)
    out = torch.zeros((D_HEAD, SIZE_M), device="cuda", dtype=torch.float16)

    grid = (1,)
    kernel[grid](
        matrix,
        out,
        *matrix.stride(),
        *out.stride(),
        SIZE_M,
        D_HEAD,
    )

    return out
