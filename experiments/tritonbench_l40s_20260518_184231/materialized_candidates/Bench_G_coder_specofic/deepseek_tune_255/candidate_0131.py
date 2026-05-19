import torch
import triton
import triton.language as tl

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Program IDs for blocks of A and B
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Pointer arithmetic for blocks of A and B
    a_block_ptr = a_ptr + (offs_am[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak)
    b_block_ptr = b_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # Load blocks of A and B
    a = tl.load(a_block_ptr, mask=(offs_am[:, None] < M) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
    b = tl.load(b_block_ptr, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
    # Dot products and accumulate
    accumulator = tl.dot(a, b)
    # Apply activation (e.g., leaky ReLU)
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    # Ensure correctness with nan and inf checks
    assert tl.all(tl.isfinite(accumulator)), "Encountered NaN or inf in Triton kernel!"
    # Write-back block of C
    c_block_ptr = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_block_ptr, accumulator, mask=c_mask)

# Triton kernel for leaky ReLU activation
@triton.jit
def leaky_relu(x):
    x = x + 1
    x = tl.where(x >= 0, x, 0.01 * x)
    return x

# Function to invoke the Triton kernel
def matmul(a, b, activation=None):
    # Ensure inputs are valid
    assert a.shape[1] == b.shape[0], "Incompatible dimensions in matmul"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    # Prepare output tensor and dimension variables
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    M, N, K = c.shape
    # Define block sizes and number of programs
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    NUM_PROGRAMS = 4
    # Invoke the Triton kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=activation,
        num_programs=NUM_PROGRAMS,
        num_stages=1,
    )
    return c
