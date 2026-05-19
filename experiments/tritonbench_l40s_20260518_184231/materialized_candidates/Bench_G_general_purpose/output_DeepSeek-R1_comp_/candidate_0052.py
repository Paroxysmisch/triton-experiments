import triton
import triton.language as tl
import torch

@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    
    group_id = tile_id // (GROUP_M * num_n_tiles)
    group_size = GROUP_M * num_n_tiles
    remainder = tile_id % group_size
    
    tile_m_in_group = remainder // num_n_tiles
    tile_n = remainder % num_n_tiles
    tile_m = group_id * GROUP_M + tile_m_in_group
    
    tile_m = tl.minimum(tile_m, num_m_tiles - 1)
    return tile_m, tile_n

@triton.jit
def linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles
    return tile_m, tile_n

@triton.jit
def mac_loop(
    A, B, C, M, N, K, locks,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    iters_per_tile, start_iter, end_iter,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    tile_m, tile_n = swizzle_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    for k_iter in range(start_iter, end_iter):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_iter * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_iter * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    offs_cm = offs_m[:, None]
    offs_cn = offs_n[None, :]
    c_ptrs = C + offs_cm * stride_cm + offs_cn * stride_cn
    
    if locks is not None:
        lock_id = tile_m * ((N + BLOCK_N - 1) // BLOCK_N) + tile_n
        while tl.atomic_cas(locks + lock_id, 0, 1) != 0:
            pass
        tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        tl.atomic_xchg(locks + lock_id, 0)
    else:
        tl.store(c_ptrs, acc.to(C.dtype.element_ty))

@triton.jit
def first_wave(
    A, B, C, M, N, K, locks,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    total_full_tiles_streamk, total_partial_tiles_streamk,
    iters_per_tile, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    total_tiles = total_full_tiles_streamk + total_partial_tiles_streamk
    tile_id = pid % total_tiles
    iter_start = (pid // total_tiles) * iters_per_tile
    iter_end = tl.minimum(iter_start + iters_per_tile, (K + BLOCK_K - 1) // BLOCK_K)
    
    mac_loop(
        A, B, C, M, N, K, locks,
        stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        iters_per_tile, iter_start, iter_end,
        BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M
    )

@triton.jit
def full_tiles(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    total_tiles_streamk: int, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    tile_id = pid
    
    tile_m, tile_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    num_k_iters = (K + BLOCK_K - 1) // BLOCK_K
    for k_iter in range(num_k_iters):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_iter * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_iter * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    offs_cm = offs_m[:, None]
    offs_cn = offs_n[None, :]
    c_ptrs = C + offs_cm * stride_cm + offs_cn * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty))

class matmul:
    def __init__(self, BLK_M: int, BLK_N: int, BLK_K: int, stages: int, warps: int, GROUP_M: int):
        self.BLK_M = BLK_M
        self.BLK_N = BLK_N
        self.BLK_K = BLK_K
        self.stages = stages
        self.warps = warps
        self.GROUP_M = GROUP_M
        
    def _call(self, A, B, C, M, N, K, grid, locks):
        def grid_fn(meta):
            return grid
        
        iters_per_tile = (K + self.BLK_K - 1) // self.BLK_K
        total_tiles = ((M + self.BLK_M - 1) // self.BLK_M) * ((N + self.BLK_N - 1) // self.BLK_N)
        total_iterations = total_tiles * iters_per_tile
        total_tiles_streamk = total_iterations // iters_per_tile
        total_partial_tiles_streamk = total_iterations % iters_per_tile
        total_full_tiles_streamk = total_tiles_streamk - total_partial_tiles_streamk
        
        if total_partial_tiles_streamk > 0:
            first_wave[grid](
                A, B, C, M, N, K, locks,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                total_full_tiles_streamk, total_partial_tiles_streamk,
                iters_per_tile,
                self.BLK_M, self.BLK_N, self.BLK_K,
                tl.float32, self.GROUP_M
            )
        
        full_tiles_grid = (total_full_tiles_streamk,)
        full_tiles[full_tiles_grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            total_tiles_streamk,
            self.BLK_M, self.BLK_N, self.BLK_K,
            tl.float32, self.GROUP_M
        )
    
    def forward(self, a, b):
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        locks = torch.zeros((M // self.BLK_M + 1) * (N // self.BLK_N + 1), device=a.device, dtype=torch.int32)
        grid = (triton.cdiv(M, self.BLK_M) * triton.cdiv(N, self.BLK_N),)
        self._call(a, b, c, M, N, K, grid, locks)
        return c
