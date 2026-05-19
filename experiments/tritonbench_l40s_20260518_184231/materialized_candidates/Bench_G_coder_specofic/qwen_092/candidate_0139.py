import triton
import triton.language as tl

# Define block sizes
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
EVEN_K = True

@triton.jit
def matmul_kernel_with_block_pointers(
    a_ptr, b_ptr, c_ptr, scales1_ptr, M, N, K, stride_a, stride_b, stride_c, stride_s1,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K
):
    pid = tl.program_id(axis=0)
    pid_m = pid // (BLOCK_N * (N // BLOCK_N))
    pid_n = pid % (BLOCK_N * (N // BLOCK_N))
    pid_k = pid_m * (BLOCK_K // (M // BLOCK_M)) + pid_n // BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        a_frag = tl.load(a_ptr + pid_m * stride_a + k * stride_a, (BLOCK_M, BLOCK_K), mask=k + BLOCK_K < K, padding=0)
        b_frag = tl.load(b_ptr + (k + pid_k) * stride_b + pid_n * stride_b, (BLOCK_K, BLOCK_N), mask=k + BLOCK_K < K, padding=0)
        acc += tl.dot(a_frag, b_frag)

    c_offset = pid_m * stride_c + pid_n * stride_c
    c_frag = tl.load(c_ptr + c_offset, (BLOCK_M, BLOCK_N), mask=True, padding=0)
    scales1_frag = tl.load(scales1_ptr + pid_m * stride_s1 + pid_n * stride_s1, (BLOCK_M, BLOCK_N), mask=True, padding=0)
    c_frag += acc * scales1_frag
    tl.store(c_ptr + c_offset, c_frag)

@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    a_ptr, b_ptr, scales1_ptr, c_ptr, M, N, K, stride_a, stride_b, stride_c, stride_s1,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K
):
    pid = tl.program_id(axis=0)
    pid_m = pid // (BLOCK_N * (N // BLOCK_N))
    pid_n = pid % (BLOCK_N * (N // BLOCK_N))
    pid_k = pid_m * (BLOCK_K // (M // BLOCK_M)) + pid_n // BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        a_frag = tl.load(a_ptr + pid_m * stride_a + k * stride_a, (BLOCK_M, BLOCK_K), mask=k + BLOCK_K < K, padding=0)
        b_frag = tl.load(b_ptr + (k + pid_k) * stride_b + pid_n * stride_b, (BLOCK_K, BLOCK_N), mask=k + BLOCK_K < K, padding=0)
        acc += tl.dot(a_frag, b_frag)

    c_offset = pid_m * stride_c + pid_n * stride_c
    c_frag = tl.load(c_ptr + c_offset, (BLOCK_M, BLOCK_N), mask=True, padding=0)
    scales1_frag = tl.load(scales1_ptr + pid_m * stride_s1 + pid_n * stride_s1, (BLOCK_M, BLOCK_N), mask=True, padding=0)
    c_frag += acc * scales1_frag
    tl.store(c_ptr + c_offset, c_frag)

class Config:
    def __init__(self, num_warps=4, num_stages=2, num_ctas=1):
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas

def int_matmul_kernel(a, b, c, config):
    M, K = a.shape
    K, N = b.shape
    stride_a = K
    stride_b = N
    stride_c = N
    stride_s1 = 0  # Not used in this kernel
    grid = ((M - 1) // BLOCK_M + 1) * ((N - 1) // BLOCK_N + 1) * config.num_ctas
    block = (BLOCK_M, BLOCK_N, config.num_warps)
    matmul_kernel_with_block_pointers[grid, block](
        a, b, c, None, M, N, K, stride_a, stride_b, stride_c, stride_s1,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K
    )

def int_scaled_matmul_kernel(a, b, scales1, c, config):
    M, K = a.shape
    K, N = b.shape
    stride_a = K
    stride_b = N
    stride_c = N
    stride_s1 = N
    grid = ((M - 1) // BLOCK_M + 1) * ((N - 1) // BLOCK_N + 1) * config.num_ctas
    block = (BLOCK_M, BLOCK_N, config.num_warps)
    scaled_matmul_kernel_with_block_pointers[grid, block](
        a, b, scales1, c, M, N, K, stride_a, stride_b, stride_c, stride_s1,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K
    )
