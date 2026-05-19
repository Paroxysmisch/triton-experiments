import triton
import triton.language as tl
from triton import Config

@triton.jit
def matmul_kernel_with_block_pointers(
    C,
    A,
    B,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):  # Kernel function
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < (K - k * BLOCK_K) if EVEN_K else None, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < (K - k * BLOCK_K) if EVEN_K else None, other=0.0
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.int32)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    C,
    A,
    B,
    M,
    N,
    K,
    scales1,
    stride_cm,
    stride_cn,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    scales1_ptrs = scales1 + offs_bn * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < (K - k * BLOCK_K) if EVEN_K else None, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < (K - k * BLOCK_K) if EVEN_K else None, other=0.0
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_bn = tl.arange(0, BLOCK_N)
    scales1_ptrs = scales1 + offs_bn * stride_bn
    scales1 = tl.load(scales1_ptrs, mask=offs_bn < N, other=0.0).to(tl.float32)
    c = accumulator.to(tl.float32) * scales1[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def int_matmul_kernel(a, b, c, config: Config):
    launch_kwargs = {"config": config}
    a_strides = a.strides
    b_strides = b.strides
    c_strides = c.strides
    M, _ = a.shape
    _, N = b.shape
    K, _ = a.shape
    assert K % 128 == 0
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    def grid(meta): return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    matmul_kernel_with_block_pointers[grid](
        c,
        a,
        b,
        M,
        N,
        K,
        c_strides[0],
        c_strides[1],
        a_strides[0],
        a_strides[1],
        b_strides[0],
        b_strides[1],
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K % BLOCK_K == 0,
        **launch_kwargs
    )


def int_scaled_matmul_kernel(a, b, scales1, c, config: Config):
    launch_kwargs = {"config": config}
    a_strides = a.strides
    b_strides = b.strides
    scales1_strides = scales1.strides
    c_strides = c.strides
    M, _ = a.shape
    _, N = b.shape
    K, _ = a.shape
    assert K % 16 == 0
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    def grid(meta): return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    scaled_matmul_kernel_with_block_pointers[grid](
        c,
        a,
