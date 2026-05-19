import triton
import triton.language as tl

# Define the Triton kernel for block-wise matrix multiplication
@triton.jit
def matmul_kernel(x_ptr, y_ptr, z_ptr, m_size, n_size, k_size, m_block_size, n_block_size, k_block_size, **meta):
    pid = tl.program_id(0)
    
    # Compute block indices
    num_m_blocks = (m_size + m_block_size - 1) // m_block_size
    num_n_blocks = (n_size + n_block_size - 1) // n_block_size
    m_block_idx = pid // num_n_blocks
    n_block_idx = pid % num_n_blocks

    # Compute the starting indices for this block
    m_start = m_block_idx * m_block_size
    n_start = n_block_idx * n_block_size

    # Initialize an accumulator for the result block
    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    # Loop over k dimension in blocks
    for k_start in range(0, k_size, k_block_size):
        # Load blocks of x and y
        x_block = tl.load(x_ptr + (m_start + tl.arange(0, m_block_size)[:, None]) * k_size + (k_start + tl.arange(0, k_block_size)), mask=(m_start + tl.arange(0, m_block_size)[:, None] < m_size) & (k_start + tl.arange(0, k_block_size) < k_size), other=0.0)
        y_block = tl.load(y_ptr + (k_start + tl.arange(0, k_block_size)[:, None]) * n_size + (n_start + tl.arange(0, n_block_size)), mask=(k_start + tl.arange(0, k_block_size)[:, None] < k_size) & (n_start + tl.arange(0, n_block_size) < n_size), other=0.0)
        
        # Perform the dot product for the current block
        z += tl.dot(x_block, y_block)

    # Store the result back to the output matrix
    tl.store(z_ptr + (m_start + tl.arange(0, m_block_size)[:, None]) * n_size + (n_start + tl.arange(0, n_block_size)), z, mask=(m_start + tl.arange(0, m_block_size)[:, None] < m_size) & (n_start + tl.arange(0, n_block_size) < n_size))

# Wrapper function to set up and launch the kernel
def matmul(x, y, m_size, n_size, k_size, m_block_size=128, n_block_size=128, k_block_size=32):
    # Initialize output matrix
    z = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)

    # Calculate the grid size
    num_m_blocks = (m_size + m_block_size - 1) // m_block_size
    num_n_blocks = (n_size + n_block_size - 1) // n_block_size
    grid = num_m_blocks * num_n_blocks

    # Launch the kernel
    matmul_kernel[grid](
        x, y, z, 
        m_size, n_size, k_size, 
        m_block_size, n_block_size, k_block_size
    )

    return z
