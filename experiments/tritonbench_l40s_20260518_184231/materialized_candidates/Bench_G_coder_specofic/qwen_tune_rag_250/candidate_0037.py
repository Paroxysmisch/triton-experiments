import torch
import triton
import triton.language as tl
import kernel_utils

def batched_vecmat(vec, mat, block_m, block_n, block_k):
    # Wrapper function for the batched vector-matrix multiplication Triton kernel
    batch, dim_m, dim_k = vec.shape
    _, dim_m_out, dim_k_out = mat.shape
    assert vec.shape[1:] == (dim_m, dim_k)
    assert mat.shape[1:] == (dim_m, dim_n, dim_k)
    assert dim_k % block_k == 0
    assert dim_m % block_m == 0
    assert dim_n % block_n == 0

    grid = (triton.cdiv(dim_m, block_m), triton.cdiv(dim_n, block_n), batch)

    vec = vec.unsqueeze(-1).expand(batch, dim_m, dim_k)
    mat = mat.unsqueeze(0).expand(batch, dim_m, dim_n, dim_k)

    output = torch.empty((batch, dim_m, dim_n), dtype=vec.dtype, device=vec.device)
    kernel_utils.launch_kernel(batched_vecmat_kernel, grid, vec, mat, output, block_m, block_n, block_k)
    return output

@triton.jit
def batched_vecmat_kernel(A, B, output, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    # Triton kernel for batched vector-matrix multiplication
    m_index = tl.program_id(axis=0)
    n_index = tl.program_id(axis=1)
    batch_index = tl.program_id(axis=2)

    output_tile = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    vec_index = m_index * block_m + tl.arange(0, block_m)[:, None]
    mat_index = n_index * block_n + tl.arange(0, block_n)[None, :]

    a = tl.load(A + batch_index * vec.strides[0] + vec_index * vec.strides[1])
    vec_a = a[:, None, None]
    vec_a = tl.broadcast_to(vec_a, (block_k, block_m, 1))
    k_blocks = B.shape[1] // block_k

    vec_b = tl.zeros((block_k, block_m, block_n), dtype=tl.float32)
    for i in range(0, k_blocks):
        b = tl.load(B + batch_index * B.stride(0) + i * block_k * B.stride(1) + mat_index * B.stride(2))
        b = b[None, :, :]
        vec_b += vec_a * b

    vec_b = tl.sum(vec_b, axis=0)
    tl.store(output + batch_index * output.stride(0) + output_tile, vec_b)
