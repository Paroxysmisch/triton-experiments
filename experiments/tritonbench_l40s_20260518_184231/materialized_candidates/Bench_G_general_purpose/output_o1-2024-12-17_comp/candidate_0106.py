import triton
import triton.language as tl
import torch

@triton.jit
def iv_dependent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stridea_m, stridea_k,
    strideb_k, strideb_n,
    stridec_m, stridec_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Broadcast dimension for K
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Condition masks to ensure valid loads
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Loop over K dimension in steps
    # Double loop over segments of size BLOCK_SIZE_K
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        # K-range mask
        mask_k = (k_block_start + offs_k) < K

        # Offsets for A and B
        a_offsets = (offs_m[:, None] * stridea_m) + ((k_block_start + offs_k[None, :]) * stridea_k)
        b_offsets = ((k_block_start + offs_k[:, None]) * strideb_k) + (offs_n[None, :] * strideb_n)

        # Different loading strategies based on 'type'
        if type == "prefetch":
            # Prefetch-like: load a chunk of data while computing the previous
            a_block = tl.load(a_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_block = tl.load(b_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        elif type == "no_prefetch":
            # Direct/no prefetch
            a_block = tl.load(a_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_block = tl.load(b_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        else:
            # Default/fallback approach
            a_block = tl.load(a_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b_block = tl.load(b_ptr + b_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Multiply the blocks and accumulate
        acc += tl.dot(a_block, b_block)

    # Write back the result
    c_offsets = offs_m[:, None] * stridec_m + offs_n[None, :] * stridec_n
    tl.store(c_ptr + c_offsets, acc, mask=mask_m[:, None] & mask_n[None, :])


def iv_dependent_matmul_wrapper(M, N, K, type_str="no_prefetch"):
    # Random input data
    a = torch.randn((M, K), dtype=torch.float32, device="cuda")
    b = torch.randn((K, N), dtype=torch.float32, device="cuda")
    c = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    # Strides
    stridea_m = a.stride(0)
    stridea_k = a.stride(1)
    strideb_k = b.stride(0)
    strideb_n = b.stride(1)
    stridec_m = c.stride(0)
    stridec_n = c.stride(1)

    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Grid configuration
    grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )

    # Number of stages based on scheduling type
    if type_str == "prefetch":
        num_stages = 3
    else:
        num_stages = 2

    # Kernel launch
    iv_dependent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stridea_m, stridea_k,
        strideb_k, strideb_n,
        stridec_m, stridec_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        type=type_str,
        num_stages=num_stages
    )

    # The result
    triton_output = c
    return triton_output
