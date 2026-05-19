import triton.language as tl
from triton import jit
from triton.testing import torch_utils

@jit
def batched_vecmat_kernel(A, B, output,
                          m_blocks: tl.constexpr, 
                          n_blocks: tl.constexpr, 
                          k_blocks: tl.constexpr, 
                          block_m: tl.constexpr, 
                          block_n: tl.constexpr, 
                          block_k: tl.constexpr):
    # Compute output tile indices.
    m_index = tl.program_id(axis=0)
    n_index = tl.program_id(axis=1)

    m_start = m_index * block_m
    n_start = n_index * block_n

    output_tile = output[m_start:m_start+block_m, n_start:n_start+block_n]

    vecmat = tl.zeros((block_m, block_n), dtype=tl.float32)

    for kc in range(0, k_blocks):
        k_start = kc * block_k

        a_tile = A[m_start:m_start+block_m, k_start:k_start+block_k]
        b_tile = B[k_start:k_start+block_k, n_start:n_start+block_n]

        # Broadcast a to match b's shape.
        a_broadcast = a_tile[:, None, :]

        # Perform vector-matrix multiplication.
        vecmat += tl.sum(a_broadcast * b_tile, axis=-2)

    # Save the result for this block to the full output.
    tl.store(output_tile, vecmat)


def batched_vecmat(A: torch_utils.TorchTensor,
                   B: torch_utils.TorchTensor,
                   m_blocks: int,
                   n_blocks: int,
                   k_blocks: int,
                   block_m: int,
                   block_n: int,
                   block_k: int):
    assert A.shape[0] % m_blocks == 0
    assert B.shape[1] % n_blocks == 0

    # Initialize tensors on the GPU.
    device = A.device
    output = torch_utils.to_torch_tensor(
        [[[[0.0]*A.shape[1] for _ in range(A.shape[0])] 
          for _ in range(B.shape[1])] 
         for _ in range(B.shape[0])], 
        device=device, dtype=torch.float32)

    m_grid = A.shape[0] // m_blocks
    n_grid = B.shape[1] // n_blocks

    # Launch the Triton kernel.
    batched_vecmat_kernel[m_grid, n_grid](
        torch_utils.to_triton_tensor(A),
        torch_utils.to_triton_tensor(B),
        torch_utils.to_triton_tensor(output),
        m_blocks,
        n_blocks,
        k_blocks,
        block_m,
        block_n,
        block_k
    )

    return output
