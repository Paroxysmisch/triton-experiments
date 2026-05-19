import torch
import triton
import triton.language as tl
from triton import Compiler
from torch._inductor.triton_heuristics import (
    pointwise,
)
from torch._inductor.utils import instance_descriptor


@triton.jit
def batched_vecmat_kernel(
    A,
    B,
    output,
    dim_m,
    dim_n,
    dim_k,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    # Triton kernel to compute batched vector-matrix product
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    output_tile = pointwise(
        (
            dim_m,
            dim_n,
        ),
        block_shape=(block_m, block_n),
        grid_shape=(
            n_index + 1,
            m_index + 1,
        ),
        group_shape=(1, 1),
    )
    # Initialize accumulator
    vecmat = tl.zeros([block_m, block_n], dtype=tl.float32)
    # Iterate over k dimension, loading appropriate blocks of A and B
    for k_block in range(0, tl.cdiv(dim_k, block_k)):

        a = tl.load(A + m_index * dim_k + k_block * block_k + tl.arange(0, block_k))
        b = tl.load(B + m_index * dim_n * dim_k + n_index * dim_k + k_block * block_k + tl.arange(0, block_k)[:, None] * dim_k + tl.arange(0, block_n)[None, :] * dim_k * dim_n)
        b = b.to(tl.float32)
        # Broadcast a to be compatible with b
        a = tl.broadcast_to(a[:, None], (block_k, block_n))
        # Perform and accumulate the dot product
        vecmat += tl.sum(a * b, 0)

    # Store the result
    tl.store(output + m_index * dim_n * dim_m + n_index * dim_m + output_tile, vecmat)


def batched_vecmat(A: torch.Tensor, B: torch.Tensor, allow_tf32: bool = False) -> torch.Tensor:
    # Function to initialize tensors and launch the Triton kernel
    device = torch.device("cuda")
    dim_m = A.shape[0]
    dim_n = B.shape[1]
    dim_k = A.shape[1]

    assert (
        dim_m % BATCHED_VECMAT_BLOCK_SIZE_M == 0
    ), f"Batch dimension must be divisible by block size (BATCHED_VECMAT_BLOCK_SIZE_M): " \
        f"{BATCHED_VECMAT_BLOCK_SIZE_M}, got {dim_m}"
    assert (
        dim_n % BATCHED_VECMAT_BLOCK_SIZE_N == 0
    ), f"Batch dimension must be divisible by block size (BATCHED_VECMAT_BLOCK_SIZE_N): " \
        f"{BATCHED_VECMAT_BLOCK_SIZE_N}, got {dim_n}"

    num_m_blocks = (
            dim_m // BATCHED_VECMAT_BLOCK_SIZE_M
    )
    num_n_blocks = (
            dim_n // BATCHED_VECMAT_BLOCK_SIZE_N
    )

    output = torch.empty([num_m_blocks, num_n_blocks, BATCHED_VECMAT_BLOCK_SIZE_M, BATCHED_VECMAT_BLOCK_SIZE_N], device=device, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16)

    kernel = instance_descriptor(batched_vecmat_kernel, A, B, output, dim_m, dim_n, dim_k, triton_meta={"signature": {4: "*i32", 5: "*i32"}})
    grid = (num_m_blocks, num_n_blocks)
    A = A.to(device, dtype=torch.float16)
    B = B.to(device, dtype=torch.float16)
    with torch.cuda.device(device):
        kernel[grid](A, B, output, dim_m, dim_n, dim_k, BATCHED_VECMAT_BLOCK_SIZE_M, BATCHED_VECMAT_BLOCK_SIZE_N, BATCHED_VECMAT_BLOCK_SIZE_K)
    return output
