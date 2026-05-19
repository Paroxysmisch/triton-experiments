import triton
import triton.language as tl

# Constants
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 8
ACC_TYPE = tl.float32

# Function to calculate 2D tile coordinates from a given linear tile ID using a swizzling pattern
@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    group_id = tile_id // (M * N // (BLOCK_M * BLOCK_N))
    group_offset = tile_id % (M * N // (BLOCK_M * BLOCK_N))
    tile_m = (group_id * GROUP_M + group_offset // (N // BLOCK_N)) * BLOCK_M
    tile_n = (group_offset % (N // BLOCK_N)) * BLOCK_N
    return tile_m, tile_n

# Function to convert a linear tile ID into 2D tile coordinates without reordering
@triton.jit
def linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    tile_m = (tile_id // (N // BLOCK_N)) * BLOCK_M
    tile_n = (tile_id % (N // BLOCK_N)) * BLOCK_N
    return tile_m, tile_n

# Function to compute a portion of the matrix multiplication for the given range of iterations
@triton.jit
def mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    pid = tl.program_id(0)
    tile_id = pid
    tile_m, tile_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    
    offs_m = tile_m + tl.arange(0, BLOCK_M)
    offs_n = tile_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    C = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    for k in range(start_iter, end_iter, BLOCK_K):
        a = tl.load(A + k * stride_ak)
        b = tl.load(B + k * stride_bk)
        acc += tl.dot(a, b)
    
    tl.atomic_add(C, acc, allow Race=True)

# Function to manage the first set of work-items executed on the hardware
@triton.jit
def first_wave(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    pid = tl.program_id(0)
    if pid < total_full_tiles_streamk:
        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, K, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

# Function to compute tiles left after the initial "first wave"
@triton.jit
def full_tiles(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, total_tiles_streamk, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M):
    pid = tl.program_id(0)
    if pid < total_tiles_streamk:
        mac_loop(A, B, C, M, N, K, locks, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, 0, K, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)

# Class to orchestrate the execution
class matmul:
    def __init__(self, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, ACC_TYPE):
        self.M = M
        self.N = N
        self.K = K
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_M = GROUP_M
        self.ACC_TYPE = ACC_TYPE

    def _call(self, A, B, C):
        # Grid setup
        total_tiles = (self.M * self.N) // (self.BLOCK_M * self.BLOCK_N)
        total_full_tiles_streamk = total_tiles // 2
        total_partial_tiles_streamk = total_tiles - total_full_tiles_streamk
        iters_per_tile = self.K // self.BLOCK_K

        # Memory allocation for locks
        locks = tl.zeros((total_tiles,), dtype=tl.int32)

        # Grid configuration
        grid = (total_tiles,)

        # First wave of work-items
        first_wave[grid](A, B, C, self.M, self.N, self.K, locks, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile, self.BLOCK_M, self.BLOCK_N, self.BLOCK_K, self.ACC_TYPE, self.GROUP_M)

        # Remaining tiles
        full_tiles[grid](A, B, C, self.M, self.N, self.K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), total_tiles, self.BLOCK_M, self.BLOCK_N, self.BLOCK_K, self.ACC_TYPE, self.GROUP_M)

    def forward(self, A, B):
        M, K = A.shape
        K, N = B.shape
        C = tl.zeros((M, N), dtype=self.ACC_TYPE)
        self._call(A, B, C)
        return C
