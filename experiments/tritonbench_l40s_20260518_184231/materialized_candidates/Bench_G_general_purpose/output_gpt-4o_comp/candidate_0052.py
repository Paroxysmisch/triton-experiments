import triton
import triton.language as tl

# Define the swizzle_tile function
@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    # Calculate 2D tile coordinates with swizzling
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
    group_size = num_tiles_m // GROUP_M
    tile_m = (tile_id // num_tiles_n) % group_size
    tile_n = tile_id % num_tiles_n
    tile_m += (tile_id // (num_tiles_n * group_size)) * group_size
    return tile_m, tile_n

# Define the linear_tile function
@triton.jit
def linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    # Calculate 2D tile coordinates without swizzling
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
    tile_m = tile_id // num_tiles_n
    tile_n = tile_id % num_tiles_n
    return tile_m, tile_n

# Define the mac_loop function
@triton.jit
def mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    pid = tl.program_id(0)
    tile_m, tile_n = linear_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    for iter in range(start_iter, end_iter):
        offs_k = iter * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_am = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        a = A[offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak]
        b = B[offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn]
        
        acc += tl.dot(a, b)
    
    offs_cm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C[offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn] = acc

# Define the first_wave function
@triton.jit
def first_wave(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, total_full_tiles_streamk, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

# Define the full_tiles function
@triton.jit
def full_tiles(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    mac_loop(A, B, C, M, N, K, None, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, 0, total_tiles_streamk, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

# Define the matmul class
class MatMul:
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M

    def _call(self, A, B, C, M, N, K):
        grid = (triton.cdiv(M, self.BLOCK_M) * triton.cdiv(N, self.BLOCK_N),)
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = B.stride()
        stride_cm, stride_cn = C.stride()
        
        total_tiles_streamk = K // self.BLOCK_K
        total_full_tiles_streamk = total_tiles_streamk
        total_partial_tiles_streamk = 0
        
        first_wave[grid](A, B, C, M, N, K, None, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_full_tiles_streamk, total_partial_tiles_streamk, total_tiles_streamk, self.BLOCK_M, self.BLOCK_N, self.BLOCK_K, C.dtype, self.GROUP_M)
        full_tiles[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, self.BLOCK_M, self.BLOCK_N, self.BLOCK_K, C.dtype, self.GROUP_M)

    def forward(self, A, B):
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        self._call(A, B, C, M, N, K)
        return C
