import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x, y, z,
    m_size, n_size, k_size,
    m_block_size, n_block_size, k_block_size,
    pid
):
    # Determine the block of the output matrix to compute
    m_block_idx = pid // (n_block_size)
    n_block_idx = pid % n_block_size

    # Compute the indices for these blocks
    m_offsets = m_block_idx * m_block_size + tl.arange(0, m_block_size)
    n_offsets = n_block_idx * n_block_size + tl.arange(0, n_block_size)

    # Compute memory offsets for blocks of x and y
    x_offsets = m_offsets[:, None] * k_size + tl.arange(0, k_block_size)
    y_offsets = (k_block_size * k_block_size) * tl.arange(0, k_block_size) + n_offsets[None, :] * k_size + tl.arange(0, k_block_size)

    # Load blocks of x and y into shared memory
    x_block = tl.load(x + x_offsets + y_offsets[None, :] * k_size)
    y_block = tl.load(y + x_offsets[None, :] * k_size + y_offsets)

    # Compute the block of the output matrix
    z_block = tl.dot(x_block, y_block, allow_tf32=True)

    # Compute memory offsets for the block of the output matrix
    z_offsets = m_offsets[:, None] * n_size + n_offsets[None, :]

    # Store the block of the output matrix
    tl.store(z + z_offsets, z_block)

def matmul(x, y):
    # Ensure x and y are 2D tensors
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("Both input tensors must be 2-dimensional")
    
    # Ensure x and y have the same number of columns
    if x.size(1) != y.size(0):
        raise ValueError("The number of columns in the first tensor must equal the number of rows in the second tensor")
    
    # Extract the sizes of the input tensors
    m, k = x.size()
    k, n = y.size()

    # Define the block sizes
    k_block_size = 128
    m_block_size = 64
    n_block_size = 128

    # Compute the number of blocks
    n_blocks = triton.cdiv(n, n_block_size)
    m_blocks = triton.cdiv(m, m_block_size)

    # Initialize an output tensor
    z = torch.empty((m, n), device=x.device, dtype=x.dtype)

    # Define the grid size
    grid = (m_blocks * n_blocks,)

    # Launch the kernel
    matmul_kernel[grid](
        x, y, z,
        m, n, k,
        m_block_size, n_block_size, k_block_size,
        pid=triton.program_id(0)
    )

    return z
