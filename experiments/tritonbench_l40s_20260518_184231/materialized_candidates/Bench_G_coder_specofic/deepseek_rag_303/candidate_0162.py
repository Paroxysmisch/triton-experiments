import torch
import triton
import triton.language as tl
from torch import Tensor

# Kernel function for matrix multiplication using Triton
@triton.jit
def matmul_kernel(
    x, y, z,
    m_size, n_size, k_size,
    m_block_size: tl.constexpr, n_block_size: tl.constexpr, k_block_size: tl.constexpr,
    m_num_warps: tl.constexpr = 8, k_num_warps: tl.constexpr = 8, n_num_warps: tl.constexpr = 8
):
    # Get program ID
    pid = tl.program_id(0)
    n_pid = tl.cdiv(n_size, n_block_size)
    k_pid = tl.cdiv(k_size, k_block_size)

    # Compute block indices
    m_block_idx = pid // n_pid
    n_block_idx = pid % n_pid

    # Define offsets for memory accesses
    x_offset = tl.arange(0, k_block_size)
    y_offset = tl.arange(0, k_block_size)
    z_offset = tl.arange(0, n_block_size * k_block_size)

    # Initialize accumulators
    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    # Loop over k dimension in blocks
    for k_block_idx in range(0, tl.cdiv(k_size, k_block_size), k_num_warps):
        # Load blocks from x and y matrices
        x_block = tl.load(x + (m_block_idx * m_block_size + x_offset[:, None]) * k_size +
                          (k_block_idx * k_block_size + x_offset[None, :])).to(tl.float32)
        y_block = tl.load(y + (k_block_idx * k_block_size + y_offset[:, None]) * n_size +
                          (n_block_idx * n_block_size + y_offset[None, :])).to(tl.float32)

        # Compute dot product
        z_block = tl.dot(x_block, y_block)

        # Accumulate results
        z += z_block
        
        # re-initialize accumulator to prevent accumulating in the wrong place.
        z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    # Store result
    x_offset = tl.arange(0, m_block_size)
    y_offset = tl.arange(0, n_block_size)
    tl.store(z + (m_block_idx * m_block_size + x_offset[:, None]) * n_size +
             (n_block_idx * n_block_size + y_offset[None, :]),
             z)


# Function to call the Triton kernel
def matmul(a: Tensor, b: Tensor):
    # Ensure contiguity of the input tensors
    assert a.is_contiguous(), "Tensor A must be contiguous"
    assert b.is_contiguous(), "Tensor B must be contiguous"

    # Extract dimensions
    m, k = a.shape
    k, n = b.shape

    # Create output tensor
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    # Define grid size
    grid = lambda META: (m * n,)

    # Launch Triton kernel
    matmul_kernel[grid](
        a, b, c,
        m, n, k,
        m_num_warps=1,
        k_num_warps=4,
        n_num_warps=4
    )
    return c
