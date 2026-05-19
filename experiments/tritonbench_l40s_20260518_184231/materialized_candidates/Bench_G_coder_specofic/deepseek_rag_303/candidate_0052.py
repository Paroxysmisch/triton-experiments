import triton
import triton.language as tl
import torch

@triton.jit()
def linear_tile(tile_id, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    return pid_m, pid_n

@triton.jit()
def swizzle_tile(tile_id, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    GROUP_M = 4
    
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m, GROUP_M)
    
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n

@triton.jit()
def mac_loop(A, B, C, M, N, K, locks,
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
             iters_per_tile,
             start_iter, end_iter,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             ACC_TYPE: tl.constexpr):
    
    tile_id = start_iter // iters_per_tile
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    k = start_iter % iters_per_tile
    for _ in range(k):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)
    if k < K:
        tl.atomic_xchg(locks + tile_id, 1)
    else:
        tl.atomic_xchg(locks + tile_id, 0)
    for current_iter in range(k, end_iter):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
        if current_iter % iters_per_tile == (iters_per_tile - 1):  # same as last iteration
            tl.store(C + rm[:, None] *
                     stride_cm + rn[None, :] * stride_cn, acc)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

@triton.jit()
def first_wave(A, B, C, M, N, K, locks,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr):
    pid = tl.program_id(0)
    start_iter = pid * total_full_tiles_streamk
    last_iter = (pid + 1) * total_full_tiles_streamk if pid != total_full_tiles_streamk else total_partial_tiles_streamk + total_full_tiles_streamk
    while start_iter < last_iter:
        end_iter = tl.minimum(start_iter + iters_per_tile, last_iter)
        mac_loop(A, B, C, M, N, K, locks,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile,
                 start_iter, end_iter,
                 BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE)
        start_iter = end_iter

@triton.jit()
def full_tiles(A, B, C,
                M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                total_tiles_streamk: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr):
    
    tile_id = tl.program_id(0) + total_tiles_streamk
    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    k = 0
    
    for _ in range(K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    
    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)


class matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        assert a.is_contiguous() and b.is_contiguous(), "non-contiguous inputs are not supported"
        M, K = a.shape
        _, N = b.shape

        BLK_M = 128
        BLK_N = 64
        BLK_K = 32

        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

        locks = torch.zeros((M * N,), device=a.device, dtype=tl.int32)

        total_tiles = triton.cdiv(M, BLK_M) * triton.c
