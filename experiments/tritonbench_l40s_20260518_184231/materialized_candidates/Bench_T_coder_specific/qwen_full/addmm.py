import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def triton_addmm_kernel(
    input, mat1, mat2, out, beta, alpha, n, m, grid_m, grid_n, TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * TILE_M + tl.arange(0, TILE_M)[:, None]
    rn = pid_n * TILE_N + tl.arange(0, TILE_N)[None, :]
    rk = tl.arange(0, TILE_K)
    input_ptrs = input + rm * n + rn
    mat1_ptrs = mat1 + rm * m + rk[None, :]
    mat2_ptrs = mat2 + rk[:, None] + rn
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(m, TILE_K)):
        input_vals = tl.load(input_ptrs, mask=(rm < n) & (rn < n), other=0.0)
        mat1_vals = tl.load(mat1_ptrs, mask=(rm < n) & (rk < m), other=0.0).to(tl.float32)
        mat2_vals = tl.load(mat2_ptrs, mask=(rk < m) & (rn < n), other=0.0).to(tl.float32)
        acc += tl.dot(mat1_vals, mat2_vals)
        input_ptrs += TILE_K
        mat1_ptrs += TILE_K
        mat2_ptrs += TILE_K
    acc = acc * alpha
    if beta != 0:
        input_ptrs = input + rm * n + rn
        input_vals = tl.load(input_ptrs, mask=(rm < n) & (rn < n), other=0.0)
        acc = acc + input_vals * beta
    tl.store(out + rm * n + rn, acc, mask=(rm < n) & (rn < n))


def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, *, beta=1, alpha=1, out=None) -> Tensor:
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert input.shape[0] == mat1.shape[0] and input.shape[1] == mat2.shape[1], "Incompatible dimensions"
    n, m = mat1.shape
    _, p = mat2.shape
    if out is None:
        out = torch.empty((n, p), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (n, p), "Out shape must be {} but got {}".format((n, p), out.shape)
    grid = lambda META: (triton.cdiv(n, META["TILE_M"]), triton.cdiv(p, META["TILE_N"]))
    triton_addmm_kernel[grid](
        input, mat1, mat2, out, beta, alpha, n, m, n // 128, p // 128
    )
    return out
