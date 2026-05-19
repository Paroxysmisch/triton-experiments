import triton
import triton.language as tl

@triton.jit
def matmul_kernel_with_block_pointers(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N

    # Initialize block pointers for A and B
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, block_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(block_m, block_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_block_ptr, boundary_check=(0, 1))
        b = tl.load(B_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a.to(tl.int32), b.to(tl.int32), allow_tf32=False)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    tl.store(C_block_ptr, accumulator.to(C_block_ptr.type.element_ty))


@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, scales1_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sm, stride_sn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N

    # Block pointers for A, B, scales, and C
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, block_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    scales_block_ptr = tl.make_block_ptr(
        base=scales1_ptr,
        shape=(M, N),
        strides=(stride_sm, stride_sn),
        offsets=(block_m, block_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(block_m, block_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_block_ptr, boundary_check=(0, 1))
        b = tl.load(B_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a.to(tl.int32), b.to(tl.int32), allow_tf32=False)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # Apply scaling
    scales = tl.load(scales_block_ptr, boundary_check=(0, 1))
    scaled_accumulator = accumulator * scales
    tl.store(C_block_ptr, scaled_accumulator.to(C_block_ptr.type.element_ty))


def int_matmul_kernel(a, b, c, config):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, config.BLOCK_M) * triton.cdiv(N, config.BLOCK_N), )
    matmul_kernel_with_block_pointers[grid](
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        BLOCK_K=config.BLOCK_K,
        EVEN_K=(K % config.BLOCK_K) == 0,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
    )


def int_scaled_matmul_kernel(a, b, scales1, c, config):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, config.BLOCK_M) * triton.cdiv(N, config.BLOCK_N), )
    scaled_matmul_kernel_with_block_pointers[grid](
        a.data_ptr(),
        b.data_ptr(),
        scales1.data_ptr(),
        c.data_ptr(),
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        scales1.stride(0), scales1.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        BLOCK_K=config.BLOCK_K,
        EVEN_K=(K % config.BLOCK_K) == 0,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
    )


class Config:
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=3, num_ctas=1):
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
