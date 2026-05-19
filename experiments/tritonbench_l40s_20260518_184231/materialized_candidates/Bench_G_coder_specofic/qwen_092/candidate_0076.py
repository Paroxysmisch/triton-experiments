import triton
import triton.language as tl

@triton.jit
@triton.autotune(
    configs=[
        triton.Config({'num_stages': 1, 'num_warps': 4}, num_stages=1, num_warps=4),
        triton.Config({'num_stages': 2, 'num_warps': 2}, num_stages=2, num_warps=2),
        triton.Config({'num_stages': 4, 'num_warps': 1}, num_stages=4, num_warps=1),
    ],
    key=['M', 'K']
)
def quantize_int8_perrow_kernel(fpa_ptr, a_ptr, as_ptr, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K):
    pid = tl.program_id(axis=0)
    block_m = pid * BLOCK_SIZE_M
    offsets = block_m * K

    fpa = tl.load(fpa_ptr + offsets, mask=block_m < M, cache=tl.CacheHint.READ_ONLY_STREAMING)
    a = tl.zeros((BLOCK_SIZE_M, K), dtype=tl.int8)
    as_out = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    max_abs = -1.0
    for k in range(0, K, BLOCK_SIZE_K):
        block_k = pid * BLOCK_SIZE_K
        fpa_block = tl.load(fpa_ptr + offsets + block_k * K, mask=(block_m < M) & (block_k < K), cache=tl.CacheHint.READ_ONLY_STREAMING)
        max_abs_block = tl.max(tl.abs(fpa_block))
        max_abs = tl.maximum(max_abs, max_abs_block)

    as_out = max_abs
    a = tl.round(fpa / max_abs * 127.0).to(tl.int8)
    tl.store(a_ptr + offsets, a, mask=block_m < M)
    tl.store(as_ptr + pid * BLOCK_SIZE_M, as_out, mask=block_m < M)

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    row = pid // grid_n
    col = pid % grid_n
    offsets_a = row * BLOCK_SIZE_M * K
    offsets_b = col * BLOCK_SIZE_N * K
    offsets_c = row * BLOCK_SIZE_M * N + col * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        block_k = pid % grid_n
        a_block = tl.load(a_ptr + offsets_a + block_k * BLOCK_SIZE_K, mask=(row < grid_m) & (block_k < grid_n), cache=tl.CacheHint.READ_ONLY_STREAMING)
        b_block = tl.load(b_ptr + offsets_b + block_k * BLOCK_SIZE_K, mask=(col < grid_n) & (block_k < grid_n), cache=tl.CacheHint.READ_ONLY_STREAMING)
        a_block_scaled = a_block * tl.load(as_ptr + row * K + block_k)
        b_block_scaled = b_block * tl.load(bs_ptr + col * K + block_k)
        accumulator += tl.dot(a_block_scaled, b_block_scaled, dtype=tl.float32)
    tl.store(c_ptr + offsets_c, accumulator, mask=(row < grid_m) & (col < grid_n))

def quantize_int8_perrow(fpa, as_ptr, M, K):
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), 1)
    quantize_int8_perrow_kernel[grid_size](fpa, as_ptr, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K)

def matmul_quantize_int8(fpa, fpb, M, K, N):
    as_ptr = tl.zeros((M,), dtype=tl.float32)
    bs_ptr = tl.zeros((N,), dtype=tl.float32)
    quantize_int8_perrow(fpa, as_ptr, M, K)
    quantize_int8_perrow(fpb, bs_ptr, K, N)
    c = tl.zeros((M, N), dtype=tl.float32)
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1)
    matmul_kernel[grid_size](fpa, fpb, c, as_ptr, bs_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K)
    return c

def quantize_int8(fpa, M, K):
    as_ptr = tl.zeros((M,), dtype=tl.float32)
    quantize_int8_perrow(fpa, as_ptr, M, K)
    a = tl.zeros((M, K), dtype=tl.int8)
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), 1)
    quantize_int8_perrow_kernel[grid_size](fpa, a, as_ptr, M, K, BLOCK_SIZE_M, BLOCK_SIZE_K)
    return a, as_ptr
