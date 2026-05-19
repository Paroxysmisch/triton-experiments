import triton
import triton.language as tl

# Define the Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Compute program IDs for each dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting offsets for each block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize the accumulator for the output matrix
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load sub-matrices from A and B
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (offs_k + k)[None, :] * stride_ak, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptr + (offs_k + k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)

        # Compute the matrix product for this block
        acc += tl.dot(a, b)

    # Write the result to the output matrix C
    c = acc.to(tl.float32)  # Ensure the accumulator is cast to the correct type
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, c, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

# Wrapper function for the matrix multiplication kernel
def matmul(a, b, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, num_stages=2, num_warps=4):
    # Ensure input matrices are compatible
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions for multiplication"

    # Extract dimensions
    M, K = a.shape
    _, N = b.shape

    # Allocate output matrix
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bn=b.stride(0),
        stride_bk=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_stages=num_stages,
        num_warps=num_warps
    )

    return c
