import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x_ptr,  # Pointer to matrix x
    y_ptr,  # Pointer to matrix y
    z_ptr,  # Pointer to output matrix z
    m_size, # Number of rows in x and z
    k_size, # Number of columns in x and rows in y
    n_size, # Number of columns in y and z
    m_block_size, 
    k_block_size, 
    n_block_size,
    **meta
):
    pid = tl.program_id(0)

    # Number of blocks along M and N dimensions
    grid_m = (m_size + m_block_size - 1) // m_block_size
    grid_n = (n_size + n_block_size - 1) // n_block_size

    # Determine which block of the output this program is responsible for
    row_block = pid // grid_n
    col_block = pid % grid_n

    # Starting offsets for the block in the output
    row_offset = row_block * m_block_size
    col_offset = col_block * n_block_size

    # Create index ranges for rows and cols within a block
    row_idx = tl.arange(0, m_block_size)
    col_idx = tl.arange(0, n_block_size)

    # Expand to 2D tile indices
    r = row_offset + row_idx[:, None]    # shape: (m_block_size, 1)
    c = col_offset + col_idx[None, :]    # shape: (1, n_block_size)

    # Initialize z_tile with zeros
    z_tile = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    # Loop over all k-blocks
    for k_offset in range(0, k_size, k_block_size):
        # Indices for sub-block of X
        k_idx_x = tl.arange(0, k_block_size)
        x_row = r                       # shape: (m_block_size, 1)
        x_col = k_offset + k_idx_x[None, :]  # shape: (1, k_block_size)

        # Indices for sub-block of Y
        k_idx_y = tl.arange(0, k_block_size)
        y_row = k_offset + k_idx_y[:, None]  # shape: (k_block_size, 1)
        y_col = c

        # Create masks to handle out-of-bounds
        x_mask = (x_row < m_size) & (x_col < k_size)
        y_mask = (y_row < k_size) & (y_col < n_size)

        # Load blocks from x and y
        x_block = tl.load(x_ptr + x_row * k_size + x_col, mask=x_mask, other=0.0)
        y_block = tl.load(y_ptr + y_row * n_size + y_col, mask=y_mask, other=0.0)

        # Accumulate partial results
        z_tile += tl.dot(x_block, y_block)

    # Write result back to global memory
    z_mask = (r < m_size) & (c < n_size)
    tl.store(z_ptr + r * n_size + c, z_tile, mask=z_mask)


def matmul(x, y, m_block_size=64, k_block_size=32, n_block_size=64):
    import triton
    import triton.language as tl
    assert x.shape[1] == y.shape[0], "Incompatible dimensions for matrix multiplication"
    m_size, k_size = x.shape
    _, n_size = y.shape

    # Allocate output
    import torch
    z = torch.empty((m_size, n_size), dtype=x.dtype, device=x.device)

    # Determine how many blocks along each dimension
    grid_m = (m_size + m_block_size - 1) // m_block_size
    grid_n = (n_size + n_block_size - 1) // n_block_size
    grid = grid_m * grid_n

    # Launch the kernel
    matmul_kernel[grid](
        x, y, z, 
        m_size, k_size, n_size, 
        m_block_size, k_block_size, n_block_size
    )
    return z
