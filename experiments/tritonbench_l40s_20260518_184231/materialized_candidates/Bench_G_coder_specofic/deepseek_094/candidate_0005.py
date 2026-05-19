import triton
import numpy as np

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_a, stride_b, stride_c, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    # Calculate the thread ids
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    thread_id = tl.thread_id()

    # Compute the offsets
    offset_a = block_id_m * BLOCK_SIZE_M * stride_a + thread_id
    offset_b = block_id_n * BLOCK_SIZE_N * stride_b
    offset_c = block_id_m * BLOCK_SIZE_M * stride_c + thread_id

    # Initialize variables
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.int32)

    # Accumulate dot products of int8 elements
    for k in range(0, K, BLOCK_SIZE_K):
        a_vals = tl.load(a_ptr + offset_a + k * stride_a)
        b_vals = tl.load(b_ptr + offset_b + k * stride_b)
        acc += a_vals * b_vals

    # Store the result
    tl.store(c_ptr + offset_c, acc)

def matmul(a, b, M, N, K):
    # Check compatibility
    assert a.dtype == np.int8
    assert b.dtype == np.uint8
    assert a.shape == (M, 4*K)
    assert b.shape == (K, N)

    # Initialize output
    c = np.zeros((M, N), dtype=np.int32)

    # Define grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))

    # Launch kernel
    matmul_kernel[grid](a, b, c, M, N, K, a.strides[0], b.strides[0], c.strides[0], BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=16, GROUP_SIZE_M=8)

    return c
