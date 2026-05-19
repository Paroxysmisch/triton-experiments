import torch
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    A,
    B,
    output,
    dim_m: tl.constexpr,
    dim_n: tl.constexpr,
    dim_k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    m_index = tl.program_id(axis=0)
    n_index = tl.program_id(axis=1)

    output_tile = tl.make_block_ptr(
        base=output,
        shape=(dim_m, dim_n),
        strides=(1, 0),
        offsets=(m_index * block_m, n_index * block_n),
        block_shape=(block_m, block_n),
        order=(1, 0),
    )

    vecmat = tl.zeros((block_m, block_n), dtype=tl.float32)
    a = tl.zeros((block_m, block_k), dtype=A.dtype.element_ty)
    b = tl.zeros((block_n, block_k), dtype=B.dtype.element_ty)

    k_blocks = dim_k // block_k

    for k in range(0, k_blocks):
        a = tl.load(
            tl.make_block_ptr(
                base=A,
                shape=(dim_m, dim_k),
                strides=(1, 0),
                offsets=(m_index * block_m, k * block_k),
                block_shape=(block_m, block_k),
                order=(1, 0),
            )
        )

        b = tl.load(
            tl.make_block_ptr(
                base=B,
                shape=(dim_m, dim_n, dim_k),
                strides=(1, 0, dim_m),
                offsets=(m_index * block_m, n_index * block_n, k * block_k),
                block_shape=(block_m, block_n, block_k),
                order=(2, 0, 1),
            )
        )

        a = tl.broadcast_to(a, (block_m, block_k))
        b = tl.broadcast_to(b, (block_k, block_n))

        vecmat += tl.sum(a * b, axis=1)

    tl.store(output_tile, vecmat.to(output.dtype.element_ty))


def batched_vecmat(vecmat_in: torch.Tensor, vecmat_vec: torch.Tensor):
    """
    Batched vector-matrix multiplication
    vecmat_in: (dim_m, dim_k) tensor
    vecmat_vec: (dim_m, dim_n, dim_k) tensor
    out: (dim_m, dim_n) tensor
    """

    batch_shape = vecmat_vec.shape[:-2]
    vecmat_vec = vecmat_vec.reshape(-1, *vecmat_vec.shape[-2:])
    vecmat_in = vecmat_in.reshape(-1, *vecmat_in.shape[-1:])
    out = torch.empty(
        (*batch_shape, vecmat_in.shape[-2], vecmat_vec.shape[-2]),
        dtype=vecmat_in.dtype,
        device=vecmat_in.device,
    )

    dim_m, dim_k = vecmat_in.shape[-2:]
    dim_m2, dim_n, dim_k2 = vecmat_vec.shape[-3:]
    assert dim_k == dim_k2
    assert dim_m == dim_m2
    assert dim_k % 32 == 0
    assert dim_n % 32 == 0

    grid = (
        triton.cdiv(dim_m, 32),
        triton.cdiv(dim_n, 32),
    )

    batched_vecmat_kernel[grid](
        vecmat_in,
        vecmat_vec,
        out,
        dim_m,
        dim_n,
        dim_k,
        32,
        32,
        32,
    )

    return out.reshape(*batch_shape, -1)
