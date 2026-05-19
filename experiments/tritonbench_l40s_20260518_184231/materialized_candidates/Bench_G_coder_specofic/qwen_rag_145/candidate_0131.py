import numpy as np
import triton.language as tl
import triton
from typing import Optional

ACTIVATION = {'leaky_relu': lambda x: x if x > 0 else 0.1*x}

def assert_valid_input(A, B, C, M, N, K):
    assert A.shape == (M, K)
    assert B.shape == (K, N)
    assert C.shape == (M, N)

@triton.jit
def matmul_kernel(C_ptr, A_ptr, B_ptr, M, N, K, stride_A, stride_B, stride_C, 
                  BLOCK_SIZE_M: tl.constexpr, 
                  BLOCK_SIZE_N: tl.constexpr, 
                  BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_A + offs_k[None, :] * stride_A
    b_ptrs = B_ptr + offs_k[:, None] * stride_B + offs_n[None, :] * stride_B
    c_ptrs = C_ptr + offs_m[:, None] * stride_C + offs_n[None, :] * stride_C

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_A
        b_ptrs += BLOCK_SIZE_K * stride_B

    c = ACTIVATION['leaky_relu'](accumulator)
    tl.store(c_ptrs, c)

def matmul(A: np.ndarray, B: np.ndarray, C: np.ndarray, 
           block_size_m: int, block_size_n: int, block_size_k: int, 
           activation: Optional[str] = 'leaky_relu'):
    assert_valid_input(A, B, C, *C.shape)

    global ACTIVATION
    ACTIVATION = {'leaky_relu': lambda x: x if x > 0 else 0.1*x} if activation else {}
    
    A_gpu = triton.tensor(A)
    B_gpu = triton.tensor(B)
    C_gpu = triton.tensor(np.zeros_like(C))

    stride_A = A.strides[0] if A.ndim > 1 else 1
    stride_B = B.strides[0] if B.ndim > 1 else 1
    stride_C = C.strides[0] if C.ndim > 1 else 1

    grid = (int(np.ceil(C.shape[0] / block_size_m)), int(np.ceil(C.shape[1] / block_size_n)))
    matmul_kernel[grid](C_gpu, A_gpu, B_gpu, *C.shape, stride_A, stride_B, stride_C, 
                        block_size_m, block_size_n, block_size_k)

    C[:] = C_gpu.numpy()
