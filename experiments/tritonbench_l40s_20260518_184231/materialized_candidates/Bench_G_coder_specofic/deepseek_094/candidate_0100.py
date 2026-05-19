import triton
import triton.language as tl

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16

@triton.autotune
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a, stride_b, stride_c,
    leaky_relu=False,
    out_ptr=None,
    **kwargs
):
    grid = lambda meta: (
        tl.grid(meta, BLOCK_SIZE_M),
        tl.grid(meta, BLOCK_SIZE_N),
        tl.grid(meta, BLOCK_SIZE_K),
    )

    a_ptrs = [A_ptr + i * stride_a for i in range(K)]
    b_ptrs = [B_ptr + i * stride_b for i in range(N)]

    for block_m, block_n, block_k in grid(M, N, K):
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(block_k.start, block_k.stop, BLOCK_SIZE_K):
            a_val = tl.load(a_ptrs[k], (block_m.start, block_k.start))
            b_val = tl.load(b_ptrs[k], (block_k.start, block_n.start))
            acc += a_val * b_val

        if leaky_relu:
            acc = tl.where(acc > 0, acc, acc * 0.01)

        tl.store(C_ptr, acc, (block_m.start, block_n.start))

def matmul(A, B, leaky_relu=False):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    N, _ = B.shape
    C = tl.zeros((M, N), dtype=tl.float32)

    stride_a = A.strides[0]
    stride_b = B.strides[0]
    stride_c = C.strides[0]

    matmul_kernel(
        A.ctypes.data,
        B.ctypes.data,
        C.ctypes.data,
        M, N, K,
        stride_a, stride_b, stride_c,
        leaky_relu=leaky_relu,
    )

    return C
