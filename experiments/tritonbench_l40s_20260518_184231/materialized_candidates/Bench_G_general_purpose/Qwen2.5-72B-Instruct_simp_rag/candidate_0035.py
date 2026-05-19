import triton
import triton.language as tl

# Triton kernel for batched vector-matrix multiplication and elementwise multiplication
@triton.jit
def batched_vecmat_kernel(C, A, B, M, N, K, batch_size,
                          stride_cm, stride_cn,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    # Triton kernel for batched vector-matrix multiplication with extra elementwise operation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (pid_b * M * stride_am + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (pid_b * N * stride_bn + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator * accumulator
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + (pid_b * M * stride_cm + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    tl.store(c_ptrs, c)

# Wrapper function to set up the inputs and the computation grid
def batched_vecmat(C, A, B, M, N, K, batch_size, BLOCK_M, BLOCK_N, BLOCK_K):
    # Compute the number of program IDs in each dimension
    grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), batch_size)

    # Launch the kernel
    batched_vecmat_kernel[grid](C, A, B, M, N, K, batch_size,
                                C.stride(0), C.stride(1),
                                A.stride(0), A.stride(1),
                                B.stride(0), B.stride(1),
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
