import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 64}, num_warps=1),
        triton.Config({"BLOCK_SIZE_N": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE_N": 256}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def _swiglu_fwd_kernel(
    X_PTR, 
    Y_PTR, 
    OUT_PTR,
    M, 
    N,
    stride_xm, 
    stride_xn,
    stride_ym, 
    stride_yn,
    stride_om, 
    stride_on,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_ids = col_block_id * BLOCK_SIZE_N + cols
    mask = col_ids < N

    x = tl.load(X_PTR + row_id * stride_xm + col_ids * stride_xn, mask=mask, other=0.0)
    y = tl.load(Y_PTR + row_id * stride_ym + col_ids * stride_yn, mask=mask, other=0.0)
    out = x / (1.0 + tl.exp(-x)) * y
    tl.store(OUT_PTR + row_id * stride_om + col_ids * stride_on, out, mask=mask)


def _swiglu_fwd(xy):
    xy = xy.contiguous()
    M, twoN = xy.shape
    N = twoN // 2
    x = xy[:, :N].contiguous()
    y = xy[:, N:].contiguous()
    out = torch.empty_like(x)
    grid = (M, triton.cdiv(N, 128))
    _swiglu_fwd_kernel[grid](
        x, y, out,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
    )
    return out
