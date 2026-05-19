import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(Out, A, B, M, N, K, stride_om, stride_on, stride_am, stride_ak, stride_bk, stride_bn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for the A and B matrices
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer arithmetic for loading data from A and B
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back the results to the output matrix
    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    o_ptrs = Out + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    tl.store(o_ptrs, accumulator)

def batched_vecmat(Out, A, B, M, N, K, block_m, block_n, block_k, grid):
    # Compute grid size based on the dimensions and block sizes
    grid_size = (grid[0], grid[1])

    # Launch the kernel
    batched_vecmat_kernel[grid_size](
        Out, A, B, M, N, K,
        Out.stride(0), Out.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k
    )
