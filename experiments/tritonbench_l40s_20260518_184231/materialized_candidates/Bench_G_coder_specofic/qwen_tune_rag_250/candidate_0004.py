import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _swiglu_fwd_kernel(
    X,
    Y,
    OUT,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton kernel for forward pass of Swiglu
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    row_start = pid * BLOCK_SIZE
    x = tl.load(X + row_start * N + cols, mask, other=0.0)
    y = tl.load(Y + row_start * N + cols, mask, other=0.0)
    x = x * tl.sigmoid(x)
    out = x * y
    tl.store(OUT + row_start * N + cols, out, mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 16}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8)
    ],
    key=["N"],
)
@triton.jit
def _swiglu_fwd_kernel(
    X,
    Y,
    OUT,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Autotuned Triton kernel for forward pass of Swiglu
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = x * tl.sigmoid(x)
    out = x * y
    tl.store(OUT + cols, out, mask=cols < N)

def _swiglu_fwd(xy, out=None):
    # Function to prepare inputs and launch Triton kernel for Swiglu forward pass
    x, y = xy.unbind(-1)
    assert x.shape == y.shape
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
    assert x.is_contiguous()
    y = y.squeeze(-1)
    assert y.is_contiguous()
    M, N = x.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE"]),)
    _swiglu_fwd_kernel[grid](x, y, out, M, N)
    return out
