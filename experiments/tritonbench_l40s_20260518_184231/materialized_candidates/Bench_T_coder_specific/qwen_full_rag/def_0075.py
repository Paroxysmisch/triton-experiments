import torch
import triton
import triton.language as tl
from fla.utils.shape_utils import volume
from fla.runtime.triton_heuristics import tile_to_block
from torch._inductor.triton_heuristics import custom_tile
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

# Kernel function: Performs the fused Cholesky solve operation
@triton.jit
def fused_cholesky_solve_kernel(
    a_ptr, b_ptr, n, k, TILE_N: tl.constexpr, TILE_K: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    a_ptrs = a_ptr + offs_n[:, None] * n + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * n + offs_k[None, :]
    a = tl.load(a_ptrs, mask=(offs_k[None, :] <= offs_n[:, None]), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[None, :] <= offs_n[:, None]), other=0.0)
    x = triton_helpers.cholesky_solve(a, b, False)
    x_ptrs = a_ptr + offs_n[:, None] * n + k * TILE_K + offs_k[None, :]
    tl.store(x_ptrs, x, mask=(offs_k[None, :] <= offs_n[:, None]))

# Wrapper function: Invokes the Triton kernel for fused Cholesky solve
def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, k = b.shape
    assert A.shape == (n, n)
    asserk A.stride(-1) == 1
    assert b.stride(-1) == 1
    c = torch.empty_like(b)
    grid = lambda META: (
        triton.cdiv(n, META["TILE_N"]),
    )
    tile_k = max(16, tile_to_block(k))
    custom_tile(
        grid,
        fused_cholesky_solve_kernel,
        [A, b, n, k],
        {
            "TILE_N": 128,
            "TILE_K": tile_k,
            "BLOCK_SIZE": 128,
            "num_warps": 4,
            "constants": {},
        },
    )
    return c
