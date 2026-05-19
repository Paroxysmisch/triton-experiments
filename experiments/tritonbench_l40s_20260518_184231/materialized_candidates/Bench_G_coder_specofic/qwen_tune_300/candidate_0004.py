import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 512}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_N": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 512}, num_stages=2, num_warps=8),
    ],
    key=["ncols"],
)
@triton.jit
def _swiglu_fwd_kernel(
    X,
    Y,
    OUT,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = rows < M
    X = tl.load(X + rows[:, None] * N)
    Y = tl.load(Y + rows[:, None] * N)
    out = X * tl.sigmoid(X) * Y
    tl.store(OUT + rows[:, None] * N, out, mask=mask)


def _swiglu_fwd(xy, y):
    x = xy[:, 0 :: 2]
    y = y[:, 0 :: 2]
    assert x.is_contiguous()
    assert y.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_N"]),)
    _swiglu_fwd_kernel[grid](x, y, out, M, N)
    return out
