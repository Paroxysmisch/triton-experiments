import triton
import triton.language as tl
import torch

@triton.jit
def batched_vecmat_kernel(
    A_ptr: ptr, B_ptr: ptr, C_ptr: ptr,
    dim_m: int32, dim_n: int32, dim_k: int32,
    block_m: int32, block_n: int32, block_k: int32,
    stride_A: int32, stride_B: int32, stride_C: int32,
    BLOCK_SIZE_M: int32, BLOCK_SIZE_N: int32, BLOCK_SIZE_K: int32
):
    # Block indices
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    # Output matrix indices
    output_tile = tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_C + tl.arange(0, BLOCK_SIZE_N) * dim_k

    # Initialize accumulators
    vecmat = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension
    for k_block in range(0, dim_k, BLOCK_SIZE_K):
        a = tl.load(A_ptr + m_index * stride_A + k_block * dim_k, mask=k_block < dim_k)
        b = tl.load(B_ptr + n_index * stride_B + k_block * dim_k, mask=k_block < dim_k)
        b = b[:, None, :]

        # Compute dot product
        vecmat += tl.sum(a * b, axis=2)

    # Store result
    tl.store(C_ptr + output_tile, vecmat, mask=(m_index < dim_m)[:, None] * (n_index < dim_n))

def batched_vecmat(A: torch.Tensor, B: torch.Tensor, block_m: int, block_n: int, block_k: int):
    # Check dimensions
    assert A.shape[0] % block_m == 0, "dim_m must be divisible by block_m"
    assert A.shape[1] % block_k == 0, "dim_k must be divisible by block_k"
    assert B.shape[0] % block_m == 0, "dim_m must be divisible by block_m"
    assert B.shape[1] % block_n == 0, "dim_n must be divisible by block_n"
    assert B.shape[2] % block_k == 0, "dim_k must be divisible by block_k"

    # Compute grid dimensions
    grid_m = A.shape[0] // block_m
    grid_n = B.shape[1] // block_n

    # Allocate output tensor
    C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

    # Launch Triton kernel
    batched_vecmat_kernel[grid_m, grid_n](A.data_ptr(), B.data_ptr(), C.data_ptr(),
                                          A.shape[0], A.shape[1], A.shape[2],
                                          block_m, block_n, block_k,
                                          A.stride(0), B.stride(0), C.stride(0),
                                          block_m, block_n, block_k)

    return C
