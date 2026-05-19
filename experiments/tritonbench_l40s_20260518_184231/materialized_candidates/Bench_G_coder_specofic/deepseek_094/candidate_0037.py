import torch as pt
import triton as tl

@tl.kernel_function
def batched_vecmat_kernel(
    A: tl.types.array(2),
    B: tl.types.array(3),
    output: tl.types.array(3),
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    m_index: tl.constexpr,
    n_index: tl.constexpr,
):
    tl.init(output)
    a = tl.load(A, m_index, 0)
    b = tl.load(B, m_index, n_index)
    vecmat = tl.mma(a, b)
    tl.store(output, m_index, n_index, vecmat)

def batched_vecmat(A, B, block_m, block_n, block_k):
    assert A.shape[0] % block_m == 0
    assert B.shape[0] % block_m == 0
    assert A.shape[1] % block_k == 0
    assert B.shape[2] % block_k == 0

    A = pt.tensor(A, device='cuda')
    B = pt.tensor(B, device='cuda')
    output = pt.zeros((A.shape[0], B.shape[1], A.shape[1]), device='cuda')

    grid_m = A.shape[0] // block_m
    grid_n = B.shape[1] // block_n
    grid_k = A.shape[1] // block_k

    for m_index in range(grid_m):
        for n_index in range(grid_n):
            for k_index in range(grid_k):
                batched_vecmat_kernel[grid_m, grid_n](
                    A, B, output, block_m, block_n, block_k, m_index, n_index
                )
    return output.cpu().numpy()
