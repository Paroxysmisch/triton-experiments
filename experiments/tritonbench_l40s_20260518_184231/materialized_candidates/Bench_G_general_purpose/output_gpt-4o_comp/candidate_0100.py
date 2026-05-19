import triton
import triton.language as tl

# Define the matrix multiplication kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        # Add more configurations if needed
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    apply_relu: tl.constexpr
):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the range of the block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of the block in A and B
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a = tl.load(a_ptrs, mask=rk[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=rk[:, None] < K, other=0.0)

        # Compute matrix multiplication for the block
        acc += tl.dot(a, b)

        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply leaky ReLU if specified
    if apply_relu:
        acc = tl.where(acc > 0, acc, 0.01 * acc)

    # Store the result
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

# Wrapper function for matrix multiplication
def matmul(a, b, apply_relu=False):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    M, K = a.shape
    _, N = b.shape

    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define execution grid
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        apply_relu=apply_relu
    )

    return c
