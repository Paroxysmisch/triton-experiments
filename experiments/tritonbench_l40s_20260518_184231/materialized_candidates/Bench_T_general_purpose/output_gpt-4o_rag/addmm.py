import triton
import triton.language as tl
import torch

@triton.jit
def addmm_kernel(
    alpha, beta, IS_BETA_ZERO: tl.constexpr,
    M, N, K,
    mat1_ptr, mat1_stride0, mat1_stride1,
    mat2_ptr, mat2_stride0, mat2_stride1,
    input_ptr, input_stride0, input_stride1,
    out_ptr, out_stride0, out_stride1,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mat1_ptrs = mat1_ptr + offs_m[:, None] * mat1_stride0 + offs_k[None, :] * mat1_stride1
    mat2_ptrs = mat2_ptr + offs_k[:, None] * mat2_stride0 + offs_n[None, :] * mat2_stride1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mat1 = tl.load(mat1_ptrs)
        mat2 = tl.load(mat2_ptrs)
        acc += tl.dot(mat1, mat2)
        mat1_ptrs += BLOCK_SIZE_K * mat1_stride1
        mat2_ptrs += BLOCK_SIZE_K * mat2_stride0

    if IS_BETA_ZERO:
        result = alpha * acc
    else:
        input_ptrs = input_ptr + offs_m[:, None] * input_stride0 + offs_n[None, :] * input_stride1
        input = tl.load(input_ptrs)
        result = alpha * acc + beta * input

    out_ptrs = out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1
    tl.store(out_ptrs, result)

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    M, K = mat1.shape
    _, N = mat2.shape

    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)

    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))

    addmm_kernel[grid](
        alpha, beta, beta == 0,
        M, N, K,
        mat1.data_ptr(), mat1.stride(0), mat1.stride(1),
        mat2.data_ptr(), mat2.stride(0), mat2.stride(1),
        input.data_ptr(), input.stride(0), input.stride(1),
        out.data_ptr(), out.stride(0), out.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=8
    )

    return out
