import torch
import triton
import triton.language as tl

@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    """ Swizzle tile coordinates to improve L2 cache performance """
    gm = GROUP_M
    gn = 1

    # Convert to 2D tile coordinates
    tile_id_x = tile_id % gn
    tile_id_y = (tile_id // gn) % gm
    tile_id_z = (tile_id // gn // gm)

    # Swizzle tiles
    if tile_id_z % 2 == 0:
        # Even: column major
        block_id_x = tile_id_x
        block_offset_x = block_id_x * BLOCK_N

        block_id_y = tile_id_y + (tile_id_z * gm)
        block_offset_y = (block_id_y // gm) * BLOCK_M * K

        coord_x = (block_offset_x + (block_id_x * BLOCK_N))
        coord_y = (block_offset_y + (block_id_y % gm) * BLOCK_M)
        coord_k = (block_id_y * BLOCK_M)

    else:
        # Odd: row major
        block_id_y = tile_id_y
        block_offset_y = block_id_y * BLOCK_M

        block_id_x = tile_id_x + (tile_id_z * gn)
        block_offset_x = (block_id_x // gn) * BLOCK_N * K

        coord_x = (block_offset_x + (block_id_x % gn) * BLOCK_N)
        coord_y = (block_offset_y + (block_id_y // gn) * BLOCK_M)
        coord_k = (block_id_x * BLOCK_N)

    return coord_x, coord_y, coord_k


@triton.jit
def linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    """ Convert tile_id into 2D tile coordinates (without swizzling) """
    gm = GROUP_M
    gn = 1

    tile_id_x = tile_id % gn
    tile_id_y = (tile_id // gn) % gm
    tile_id_z = (tile_id // gn // gm)

    block_id_x = tile_id_x
    block_offset_x = block_id_x * BLOCK_N
    coord_x = (block_offset_x + (block_id_x * BLOCK_N))

    block_id_y = tile_id_y + (tile_id_z * gm)
    block_offset_y = (block_id_y // gm) * BLOCK_M * K
    coord_y = (block_offset_y + (block_id_y % gm) * BLOCK_M)
    coord_k = (block_id_y * BLOCK_M)

    return coord_x, coord_y, coord_k


@triton.jit
def mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    """ Loop over iterations (chunks of blocks) within a tile """
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Tile indices
    pid = tl.program_id(0)
    coord_x, coord_y, coord_k = swizzle_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # Block strides
    offs_am = coord_x + tl.arange(0, BLOCK_M)
    offs_bn = coord_y + tl.arange(0, BLOCK_N)
    offs_k = coord_k + tl.arange(0, BLOCK_K)

    # Block offsets
    block_offset_am = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    block_offset_bn = offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
    block_offset_k = offs_k * stride_k

    # Loop over K
    for k in range(start_iter, end_iter, BLOCK_K):
        # Update locks
        if k == start_iter:
            tl.store(locks + pid, 1)

        # pointers
        a_ptrs = A + block_offset_am + k * stride_k
        b_ptrs = B + block_offset_bn + k * stride_k

        # accumulate
        acc += tl.dot(a_ptrs, b_ptrs)

    # Write back
    if end_iter == iters_per_tile:
        c_ptrs = C + block_offset_am + block_offset_bn
        tl.store(c_ptrs, acc)


@triton.jit
def first_wave(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    """ First wave of tiles, which may be smaller than the others """
    pid = tl.program_id(0)

    # Full tiles
    if pid < total_full_tiles_streamk:
        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

    # Partial tiles
    else:
        partial_tile_id = pid - total_full_tiles_streamk
        num_partial_tiles = total_partial_tiles_streamk
        iters_per_tile = iters_per_tile // num_partial_tiles

        start_k = partial_tile_id * iters_per_tile

        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_k, start_k + iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)


@triton.jit
def full_tiles(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    """ Full tiles (all blocks of the tile fit in L1) """
    pid = tl.program_id(0)
    coord_x, coord_y, coord_k = linear_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # Create offsets
    offs_am = coord_x + tl.arange(0, BLOCK_M)
    offs_bn = coord_y + tl.arange(0, BLOCK_N)
    offs_k = coord_k + tl.arange(0, BLOCK_K)

    # Create pointers
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk

    # Create accumulation buffer
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Compute
        acc += tl.dot(a_ptrs, b_ptrs)

        # Advance the ptrs
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back
    offs_cm = coord_x + tl.arange(0, BLOCK_M)
    offs_cn = coord_y + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc)


class matmul(torch.autograd.Function):
    @staticmethod
    def _call(a, b, trans_a, trans_b, trans_c, M, N, K, BLK_M, BLK_N, BLK_K, num_stages, num_warps, dot_type, group_size):
        device = a.device
        # Strides
        stride_am = a.stride(0) if not trans_a else a.stride(1)
        stride_ak = a.stride(1) if not trans_a else a.stride(0)
        stride_bk = b.stride(0) if not trans_b else b.stride(1)
        stride_bn = b.stride(1) if not trans_b else b.stride(0)
        stride_cm = device._get_tensor_strides(a, (M, N))[0] if not trans_c else device._get_tensor_strides(a, (N, M))[0]
        stride_cn = device._get_tensor_strides(a, (M, N))[1] if not trans_c else device._get_tensor_strides(a, (N, M))[1]

        # Make sure we don't have an in-place operation
        assert (a is not c) and (b is not c)

        # Allocate output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        # stride_cm = c.stride(0) if not trans_c else c.stride(1)
        # stride_cn = c.stride(1) if not trans_c else c.stride(0)

        # locks for tiles
        locks
