import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def iv_dependent_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type: tl.constexpr):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the start of the block for each dimension
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulation
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks of a and b
        if type == 'pre_load':
            a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak), mask=offs_m[:, None] < M)
            b = tl.load(b_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn), mask=offs_n[None, :] < N)
        elif type == 'post_load':
            a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak))
            b = tl.load(b_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn))
        else:
            raise ValueError(f"Unsupported type: {type}")

        # Matrix multiplication
        acc += tl.dot(a, b)

    # Store the result
    c = acc.to(tl.float32)
    c_ptr += offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Define the wrapper function
def iv_dependent_matmul_wrapper(a, b, c, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, type='pre_load'):
    # Determine the number of blocks
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    iv_dependent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        type=type
    )

# Example usage:
# Assume a, b, c are Triton-managed tensors with appropriate strides
# iv_dependent_matmul_wrapper(a, b, c, M, N, K)
