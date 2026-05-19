import torch
import triton
import triton.language as tl

# Triton kernel for batched vector-matrix multiplication
@triton.jit
def batched_vecmat_kernel(
    output, A, B, dim_m, dim_n, dim_k,
    stride_am, stride_ak,
    stride_bm, stride_bn, stride_bk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    # Compute the block indices for the current program instance
    offs_m = m_index * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_index * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Compute the output tile indices
    output_tile = output + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    # Initialize the accumulator
    vecmat = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k_block in range(0, tl.cdiv(dim_k, BLOCK_K)):
        # Compute the offsets for the current K block
        a_ptrs = A + (offs_m[:, None] * stride_am + (k_block * BLOCK_K + offs_k[None, :]) * stride_ak)
        b_ptrs = B + (offs_m[:, None] * stride_bm + (k_block * BLOCK_K + offs_k[None, :]) * stride_bk + offs_n[None, :] * stride_bn)

        # Load the current block of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < dim_k - k_block * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < dim_k - k_block * BLOCK_K, other=0.0)

        # Compute the dot product for the block
        vecmat += tl.sum(a * b, axis=1)

    # Store the result for the block
    tl.store(output_tile, vecmat)

# Wrapper function to initialize tensors and launch the kernel
def batched_vecmat(A, B, output, block_m, block_n, block_k):
    dim_m, dim_k = A.shape
    dim_m, dim_n, dim_k = B.shape

    # Check that the dimensions are divisible by the block sizes
    assert dim_m % block_m == 0, "dim_m must be divisible by block_m"
    assert dim_n % block_n == 0, "dim_n must be divisible by block_n"
    assert dim_k % block_k == 0, "dim_k must be divisible by block_k"

    # Compute the grid dimensions
    grid_m = dim_m // block_m
    grid_n = dim_n // block_n

    # Launch the kernel
    batched_vecmat_kernel[grid_m, grid_n](
        output, A, B, dim_m, dim_n, dim_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k
    )

# Example usage
if __name__ == "__main__":
    # Define the dimensions
    dim_m = 128
    dim_n = 128
    dim_k = 128

    # Define the block sizes
    block_m = 16
    block_n = 16
    block_k = 16

    # Initialize the tensors
    A = torch.randn(dim_m, dim_k, device="cuda")
    B = torch.randn(dim_m, dim_n, dim_k, device="cuda")
    output = torch.zeros(dim_m, dim_n, device="cuda")

    # Perform the batched vector-matrix multiplication
    batched_vecmat(A, B, output, block_m, block_n, block_k)

    # Print the result
    print(output)
