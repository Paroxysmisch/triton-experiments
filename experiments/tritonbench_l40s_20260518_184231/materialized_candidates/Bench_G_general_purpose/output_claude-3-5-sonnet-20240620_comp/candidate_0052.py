import triton
import triton.language as tl
import torch

@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    # Calculate number of tiles in each dimension
    tiles_m = triton.cdiv(M, BLOCK_M)
    tiles_n = triton.cdiv(N, BLOCK_N)
    
    # Extract tile coordinates using bit manipulation for better cache locality
    tile_m = (tile_id // GROUP_M) * GROUP_M + (tile_id % GROUP_M)
    tile_n = tile_id // tiles_m
    
    return tile_m, tile_n

@triton.jit
def linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    tiles_m = triton.cdiv(M, BLOCK_M)
    return tile_id % tiles_m, tile_id // tiles_m

@triton.jit
def mac_loop(A, B, C, M, N, K, locks, 
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            iters_per_tile, start_iter, end_iter,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate tile coordinates
    tile_m, tile_n = swizzle_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Pointers to current tile
    offs_am = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Mask for bounds checking
    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N
    
    # Main computation loop
    for k in range(start_iter, end_iter):
        k_idx = k * BLOCK_K
        
        # Load tiles from A and B
        a = tl.load(A + offs_am[:, None] * stride_am + k_idx * stride_ak,
                   mask=a_mask[:, None], other=0)
        b = tl.load(B + k_idx * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=b_mask[None, :], other=0)
        
        # Compute matrix multiplication for current K-tile
        acc += tl.dot(a, b)
    
    # Store results
    offs_cm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Acquire lock
    lock_id = tile_m * triton.cdiv(N, BLOCK_N) + tile_n
    while tl.atomic_cas(locks + lock_id, 0, 1) == 1:
        pass
    
    # Accumulate results
    c = tl.load(C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                mask=c_mask, other=0)
    c = c + acc
    tl.store(C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=c_mask)
    
    # Release lock
    tl.atomic_xchg(locks + lock_id, 0)

@triton.jit
def first_wave(A, B, C, M, N, K, locks,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid < total_full_tiles_streamk:
        mac_loop(A, B, C, M, N, K, locks,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                iters_per_tile, 0, iters_per_tile,
                BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)
    elif pid < (total_full_tiles_streamk + total_partial_tiles_streamk):
        partial_iters = (K % (iters_per_tile * BLOCK_K)) // BLOCK_K
        mac_loop(A, B, C, M, N, K, locks,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                iters_per_tile, 0, partial_iters,
                BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

@triton.jit
def full_tiles(A, B, C, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               total_tiles_streamk,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid >= total_tiles_streamk:
        iters = K // BLOCK_K
        mac_loop(A, B, C, M, N, K, None,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                iters, 0, iters,
                BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

class matmul:
    def __init__(self, BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, GROUP_M=8,
                 num_stages=3, num_warps=8):
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M
        self.num_stages = num_stages
        self.num_warps = num_warps
        
    def _call(self, a, b):
        # Extract matrix dimensions
        M, K = a.shape
        K, N = b.shape
        
        # Allocate output
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        # Compute grid size
        tiles_m = triton.cdiv(M, self.BLOCK_M)
        tiles_n = triton.cdiv(N, self.BLOCK_N)
        total_tiles = tiles_m * tiles_n
        
        # Launch kernels
        def grid(meta):
            return (total_tiles,)
        
        # Allocate locks for synchronization
        locks = torch.zeros(total_tiles, dtype=torch.int32, device=a.device)
        
        # Launch first wave
        first_wave_kernel = first_wave[grid](
            a, b, c, M, N, K, locks,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            total_tiles // 2, total_tiles - (total_tiles // 2),
            K // self.BLOCK_K,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            a.dtype, self.GROUP_M,
            num_stages=self.num_stages,
            num_warps=self.num_warps
        )
        
        # Launch remaining tiles
        full_tiles_kernel = full_tiles[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            total_tiles // 2,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            a.dtype, self.GROUP_M,
            num_stages=self.num_stages,
            num_warps=self.num_warps
        )
        
        return c
    
    def forward(self, a, b):
        return self._call(a, b)
