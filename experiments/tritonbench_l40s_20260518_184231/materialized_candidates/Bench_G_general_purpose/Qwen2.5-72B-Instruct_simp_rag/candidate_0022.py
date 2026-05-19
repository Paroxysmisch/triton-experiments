import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    stride_am, stride_ak, stride_ao,
    stride_bk, stride_bn, stride_bo,
    stride_om, stride_on, stride_oo,
    M, N, K,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_warp = BLOCK_M * BLOCK_N // 32
    pid_m = pid // (num_pid_n * num_pid_in_warp)
    pid_n = (pid % (num_pid_n * num_pid_in_warp)) // num_pid_in_warp
    pid_k = (pid % num_pid_in_warp) * (BLOCK_K // TILE_K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, TILE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    o_ptrs = O_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(o_ptrs, accumulator)

import torch
from typing import Optional, Tuple

def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    TILE_M: int = 64,
    TILE_N: int = 64,
    TILE_K: int = 32,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
):
    assert A.is_cuda and B.is_cuda, "Tensors must be on GPU"
    assert A.dim() == 3 and B.dim() == 3, "Tensors must be 3D"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    assert A.shape[2] == B.shape[1], "Matrix dimensions must be compatible for multiplication"

    batch, M, K = A.shape
    _, _, N = B.shape

    if out is None:
        out = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)

    assert out.shape == (batch, M, N), "Output tensor shape mismatch"

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * triton.cdiv(K, BLOCK_K), batch, 1)
    if max_grid is not None:
        grid = (min(grid[0], max_grid[0]), min(grid[1], max_grid[1]), min(grid[2], max_grid[2]))

    bmm_kernel[grid](
        A, B, out,
        A.stride(1), A.stride(2), A.stride(0),
        B.stride(1), B.stride(2), B.stride(0),
        out.stride(1), out.stride(2), out.stride(0),
        M, N, K,
        TILE_M, TILE_N, TILE_K,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return out
