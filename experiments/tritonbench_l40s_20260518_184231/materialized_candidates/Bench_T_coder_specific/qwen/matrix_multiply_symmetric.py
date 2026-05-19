import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    A_shape, B_shape, C_shape,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(C_shape[0], BLOCK_SIZE_M)
    grid_n = tl.cdiv(C_shape[1], BLOCK_SIZE_N)

    m = pid // grid_n
    n = pid % grid_n

    a_ptrs = A_ptr + m * A_shape[1] * BLOCK_SIZE_K + \
             tl.arange(0, BLOCK_SIZE_M).unsqueeze(-1) * BLOCK_SIZE_K + \
             tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = B_ptr + n * B_shape[1] * BLOCK_SIZE_K + \
             tl.arange(0, BLOCK_SIZE_N).unsqueeze(-1) * BLOCK_SIZE_K + \
             tl.arange(0, BLOCK_SIZE_K)
    c_ptr = C_ptr + m * C_shape[1] * BLOCK_SIZE_N + \
           tl.arange(0, BLOCK_SIZE_M).unsqueeze(-1) * BLOCK_SIZE_N + \
           tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(tl.cdiv(B_shape[0], BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(a_ptrs < A_ptr + A_shape[0] * A_shape[1]), other=0.0)
        b = tl.load(b_ptrs, mask=(b_ptrs < B_ptr + B_shape[0] * B_shape[1]), other=0.0)
        acc += a @ b
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    tl.store(c_ptr, acc, mask=(c_ptr < C_ptr + C_shape[0] * C_shape[1]))

@triton.jit
def sym_update_kernel(
    C_ptr,
    C_shape,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(C_shape[0], BLOCK_SIZE_M)
    grid_n = tl.cdiv(C_shape[1], BLOCK_SIZE_N)

    m = pid // grid_n
    n = pid % grid_n

    c_ptr = C_ptr + m * C_shape[1] * BLOCK_SIZE_N + \
           tl.arange(0, BLOCK_SIZE_M).unsqueeze(-1) * BLOCK_SIZE_N + \
           tl.arange(0, BLOCK_SIZE_N)

    c = tl.load(c_ptr, mask=(c_ptr < C_ptr + C_shape[0] * C_shape[1]), other=0.0)
    ct = tl.transpose(c)
    acc = alpha * c @ ct + beta * c

    tl.store(c_ptr, acc, mask=(c_ptr < C_ptr + C_shape[0] * C_shape[1]))

def matrix_multiply_symmetric(A, B, C, alpha, beta):
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == C.shape[0]
    assert B.shape[1] == C.shape[1]

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Allocate memory for temporary storage
    C_temp = torch.empty_like(C)

    # Perform matrix multiplication
    matmul_kernel[
        grid=tuple(triton.div_ceil(dim, BLOCK_SIZE_M * BLOCK_SIZE_N) for dim in C.shape),
        block=(BLOCK_SIZE_M, BLOCK_SIZE_N, 1),
        num_warps=8,
    ](
        A.data_ptr(), B.data_ptr(), C_temp.data_ptr(),
        A.shape, B.shape, C.shape,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    # Perform symmetric update
    sym_update_kernel[
        grid=tuple(triton.div_ceil(dim, BLOCK_SIZE_M * BLOCK_SIZE_N) for dim in C_temp.shape),
        block=(BLOCK_SIZE_M, BLOCK_SIZE_N, 1),
        num_warps=8,
    ](
        C_temp.data_ptr(),
        C_temp.shape,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )

    return C_temp
