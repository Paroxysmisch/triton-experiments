import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # Kernel for batched matrix multiplication
    pid = tl.program_id(0)
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    group_id = tl.program_id(1)
    num_groups_m = (grid_m + GROUP_M - 1) // GROUP_M

    offs_am = (group_id * GROUP_M + tl.arange(0, TILE_M)) % grid_m
    offs_bn = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    a_ptrs = A + (
        pid // num_groups_m * M * K + (offs_am[:, None] * TILE_M + offs_k[None, :]) * K
        + offs_k[None, :]
    )
    b_ptrs = B + pid * K * N + (offs_k[:, None] * N + offs_bn[None, :]) * K + offs_bn[
        None, :
    ]

    tiles_a = tl.load(
        a_ptrs,
        mask=(offs_am[:, None] < grid_m) & (offs_k[None, :] < grid_n),
        other=0.0,
    )
    tiles_b = tl.load(
        b_ptrs,
        mask=(offs_k[:, None] < grid_k) & (offs_bn[None, :] < grid_n),
        other=0.0,
    )

    o_ptrs = O + (
        pid
        * M * N
        + (offs_am[:, None] * TILE_M + offs_bn[None, :]) * M
        + offs_bn[None, :]
    )
    o_mask = (offs_am[:, None] < grid_m) & (offs_bn[None, :] < grid_n)

    # mask_tiles_a = offs_am < grid_m
    # mask_tiles_b = offs_bn < grid_n
    # mask_a = offs_k < grid_k
    # mask_b = offs_k < grid_k
    # mask_o = mask_tiles_a & mask_tiles_b

    acc = tl.dot(tiles_a, tiles_b)
    # acc = tl.where(mask_o, acc, 0)
    tl.store(o_ptrs, acc, mask=o_mask)


def bmm(a, b):
    batch, M, K = a.shape
    _, K, N = b.shape
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        GROUP_M = META["GROUP_M"]
        num_warps = META["num_warps"]
        GROUP_N = META["GROUP_N"]
        num_stages = META["num_stages"]
        num_warps_stage = num_warps // num_stages

        grid_m = (M + META["TILE_M"] - 1) // META["TILE_M"]
        grid_n = (N + META["TILE_N"] - 1) // META["TILE_N"]
        grid_k = (K + META["TILE_K"] - 1) // META["TILE_K"]

        num_groups_m = (grid_m + GROUP_M - 1) // GROUP_M
        num_groups_n = (grid_n + GROUP_N - 1) // GROUP_N

        return (
            batch * grid_m * grid_n,
            num_warps_stage,
            num_groups_m * num_groups_n,
        )

    bmm_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        GROUP_M=4,
        GROUP_N=4,
        DIVISIBLE_M=True,
        DIVISIBLE_N=True,
        DIVISIBLE_K=True,
        num_warps=4,
        num_stages=2,
    )
    return out
