import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    A, B, output,  # Pointers to matrices
    dim_m, dim_n, dim_k,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix A
    stride_bmk, stride_bn, stride_bkn,  # Strides for matrix B
    stride_om, stride_on,  # Strides for output matrix
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr
):
    # Compute program ID in a 2D grid
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    # Compute the block indices
    rm = m_index * block_m
    rn = n_index * block_n

    # Initialize the output tile
    output_tile = tl.zeros((block_m, block_n), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k_index in range(0, dim_k, block_k):
        # Compute the block indices for the K dimension
        rk = k_index

        # Load the slices of A and B
        a = tl.load(A + rm * stride_am + rk * stride_ak, mask=rm + tl.arange(0, block_m) < dim_m, other=0.0)
        b = tl.load(B + rm * stride_bmk + rn * stride_bn + rk * stride_bkn, 
                    mask=(rm + tl.arange(0, block_m) < dim_m) & (rn + tl.arange(0, block_n) < dim_n), other=0.0)

        # Broadcast a to match the shape of b
        a = tl.broadcast_to(a, (block_m, block_n))

        # Compute the dot product for the block
        output_tile += tl.sum(a * b, axis=2)

    # Store the result in the output matrix
    tl.store(output + rm * stride_om + rn * stride_on, output_tile, mask=(rm + tl.arange(0, block_m) < dim_m) & (rn + tl.arange(0, block_n) < dim_n))

import torch

def batched_vecmat(A, B, block_m=16, block_n=16, block_k=16):
    # Get the dimensions of the input tensors
    dim_m, dim_k = A.shape
    dim_m, dim_n, dim_k = B.shape

    # Check that the dimensions are divisible by the block sizes
    assert dim_m % block_m == 0, "dim_m must be divisible by block_m"
    assert dim_n % block_n == 0, "dim_n must be divisible by block_n"
    assert dim_k % block_k == 0, "dim_k must be divisible by block_k"

    # Initialize the output tensor
    output = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)

    # Compute the grid dimensions
    grid = (dim_m // block_m, dim_n // block_n)

    # Launch the Triton kernel
    batched_vecmat_kernel[grid](
        A, B, output,
        dim_m, dim_n, dim_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        output.stride(0), output.stride(1),
        block_m, block_n, block_k
    )

    return output
