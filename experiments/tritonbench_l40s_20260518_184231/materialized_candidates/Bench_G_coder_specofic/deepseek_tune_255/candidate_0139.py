import triton
import triton.language as tl

# Kernel function with block pointers for integer matrix multiplication
@triton.jit
def matmul_kernel_with_block_pointers(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_block_ptr = tl.make_block_ptr(a_ptr, (K, M), (stride_ak, stride_am), (rk, ram), (EVEN_K, 1), (1, 0))
    b_block_ptr = tl.make_block_ptr(b_ptr, (K, N), (stride_bk, stride_bn), (rk, rbn), (EVEN_K, 1), (0, 1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b)
        if not EVEN_K:
            k += BLOCK_K
            a_block_ptr = tl.advance(a_block_ptr, [BLOCK_K, 0])
            b_block_ptr = tl.advance(b_block_ptr, [0, BLOCK_K])
    c = acc.to(tl.float32)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptrs, c, mask=c_mask)

# Kernel function with block pointers for scaled integer matrix multiplication
@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    a_ptr, b_ptr, scales1_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    first_pid_n = group_id * num_pid_n
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = first_pid_n + (pid % num_pid_n)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_block_ptr = tl.make_block_ptr(a_ptr, (K, M), (stride_ak, stride_am), (rk, ram), (EVEN_K, 1), (1, 0))
    b_block_ptr = tl.make_block_ptr(b_ptr, (K, N), (stride_bk, stride_bn), (rk, rbn), (EVEN_K, 1), (0, 1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b)
        if not EVEN_K:
            k += BLOCK_K
            a_block_ptr = tl.advance(a_block_ptr, [BLOCK_K, 0])
            b_block_ptr = tl.advance(b_block_ptr, [0, BLOCK_K])
    c = acc.to(tl.float32)
    scales1 = tl.load(scales1_ptr + rbn[None, :])
    c = c * scales1
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptrs, c, mask=c_mask)

# Function to launch the kernel
def int_matmul_kernel(a, b, c, config):
    a_np = a.data_ptr
    b_np = b.data_ptr
    c_np = c.data_ptr
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
