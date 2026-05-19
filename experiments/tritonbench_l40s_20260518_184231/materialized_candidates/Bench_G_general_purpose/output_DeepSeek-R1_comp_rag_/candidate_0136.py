import triton
import triton.language as tl

@triton.jit
def squared_matmul_kernel_with_block_pointers(
    # Pointers to matrices
    C_ptr, A_ptr, B_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride variables
    stride_Cm, stride_Cn,
    stride_Am, stride_Ak,
    stride_Bk, stride_Bn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    # Other configurable parameters
    NUM_SM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create block pointers for A and B
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_Am, stride_Ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_Bk, stride_Bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    if EVEN_K:
        for _ in range(K // BLOCK_K):
            a = tl.load(A_block_ptr)
            b = tl.load(B_block_ptr)
            accumulator += tl.dot(a, b, allow_tf32=False)
            A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
    else:
        for k in range(0, K, BLOCK_K):
            a = tl.load(A_block_ptr, boundary_check=(1,), padding_option="zero")
            b = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            accumulator += tl.dot(a, b, allow_tf32=False)
            A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # Square the accumulated result element-wise
    c = accumulator * accumulator

    # Create block pointer for C and store the result
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_Cm, stride_Cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    tl.store(C_block_ptr, c.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))

def int_squared_matmul_kernel(a, b, c, config):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, config.BLOCK_M) * triton.cdiv(N, config.BLOCK_N), )
    EVEN_K = (K % config.BLOCK_K == 0)
    squared_matmul_kernel_with_block_pointers[grid](
        c, a, b,
        M, N, K,
        c.stride(0), c.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        BLOCK_K=config.BLOCK_K,
        EVEN_K=EVEN_K,
        NUM_SM=config.num_ctas,
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
