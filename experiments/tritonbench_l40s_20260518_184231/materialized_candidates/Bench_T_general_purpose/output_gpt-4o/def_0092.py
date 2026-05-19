import torch
import triton
import triton.language as tl

# Triton kernel to compute tensordot and reciprocal square root
@triton.jit
def tensordot_rsqrt_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Create program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define offsets for a and b
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks of a and b
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (offs_k + k) * stride_ak))
        b = tl.load(b_ptr + ((offs_k + k)[:, None] * stride_bk + offs_n * stride_bn))

        # Accumulate product
        acc += tl.dot(a, b)

    # Compute reciprocal square root
    result = tl.rsqrt(acc)

    # Store result
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n * stride_cn), result)

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Perform tensordot using PyTorch to get the shape information
    c = torch.tensordot(a, b, dims=dims)

    # Get the dimensions for the Triton kernel
    M, N = c.shape
    K = a.shape[dims[0][0]]  # Assuming dims is a tuple of lists

    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Allocate output tensor
    c_triton = torch.empty_like(c)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    tensordot_rsqrt_kernel[grid](
        a, b, c_triton,
        M, N, K,
        a.stride(0), a.stride(dims[0][0]),
        b.stride(dims[1][0]), b.stride(1),
        c_triton.stride(0), c_triton.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return c_triton
