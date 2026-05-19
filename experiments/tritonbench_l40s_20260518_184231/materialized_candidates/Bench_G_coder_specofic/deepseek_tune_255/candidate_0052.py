import torch
import triton
import triton.language as tl

@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # Swizzle the tiles in a 2D grid to improve L2 cache performance
    n_tiles_per_group = (N + BLOCK_N - 1) // BLOCK_N
    group_id = tile_id // n_tiles_per_group
    n_tiles = min(n_tiles_per_group, (N + BLOCK_N - 1) // BLOCK_N - tile_id)
    m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    # Vertical swizzling
    tile_id = tile_id + (group_id % 2) * n_tiles
    new_tile_id = tile_id % n_tiles
    tile_id = tile_id - new_tile_id
    # Horizontal swizzling
    new_tile_id = new_tile_id + (group_id // 2) * m_tiles
    tile_id = tile_id + new_tile_id
    return tile_id

@triton.jit
def linear_tile(tile_id, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # Convert a linear tile ID into 2D tile coordinates without reordering
    n_tiles_per_group = (N + BLOCK_N - 1) // BLOCK_N
    group_id = tile_id // n_tiles_per_group
    n_tiles = min(n_tiles_per_group, (N + BLOCK_N - 1) // BLOCK_N - tile_id)
    m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    # Vertical swizzling
    tile_id = tile_id + (group_id % 2) * n_tiles
    new_tile_id = tile_id % n_tiles
    tile_id = tile_id - new_tile_id
    # Horizontal swizzling
    new_tile_id = new_tile_id + (group_id // 2) * m_tiles
    tile_id = tile_id + new_tile_id
    return tile_id

@triton.jit
def mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    # Compute a portion of the matmul
    pid = tl.program_id(0)
    tile_id = tl.program_id(1)
    k_tile_offsets = (tile_id % iters_per_tile) * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_tile_offsets < K
    if ACC_TYPE == "fp32":
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    elif ACC_TYPE == "fp16":
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    m_offsets = (tile_id // iters_per_tile) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = (tile_id % iters_per_tile) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    a_ptrs = A + (m_offsets[:, None] * stride_am + k_tile_offsets[None, :] * stride_ak)
    b_ptrs = B + (k_tile_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn)
    for k in range(start_iter, end_iter):
        if k_mask[k]:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
            accumulator += tl.dot(a, b)
    c_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def first_wave(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    # Manage the first set of work-items executed on the hardware
    pid = tl.program_id(0)
    tile_id = tl.program_id(1)
    full_tiles = total_full_tiles_streamk * iters_per_tile
    if tile_id < full_tiles:
        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)
    elif tile_id == full_tiles:
        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, total_partial_tiles_streamk, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

@triton.jit
def full_tiles(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    # Compute tiles left after the initial "first wave"
    pid = tl.program_id(0)
    tile_id = tl.program_id(1)
    iters_per_tile = (BLOCK_M * BLOCK_N) // (8 * 1024)
    if tile_id < total_tiles_streamk:
        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, iters_per_tile, BLOCK_M
