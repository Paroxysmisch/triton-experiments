import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, out_stride):
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    sum = 0.0
    for k in range(K):
        A_val = tl.load(A_ptr + row * out_stride + k)
        B_val = tl.load(B_ptr + k * out_stride + col)
        sum += A_val * B_val
    tl.store(C_ptr + row * out_stride + col, sum)

@triton.autotune
def matmul(A, B, C):
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == C.shape[0]
    assert B.shape[1] == C.shape[1]
    grid = lambda M, N: (M, N)
    matmul_kernel[grid](A, B, C, A.shape[0], A.shape[1], B.shape[1], C.shape[1])
