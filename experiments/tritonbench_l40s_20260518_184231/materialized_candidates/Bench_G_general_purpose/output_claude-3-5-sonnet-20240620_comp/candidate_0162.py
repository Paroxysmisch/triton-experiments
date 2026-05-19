import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    # Pointers to matrices in global memory
    x_ptr, y_ptr, z_ptr,
    # Matrix dimensions
    m_size, n_size, k_size,
    # Block sizes for tiling
    m_block_size: tl.constexpr, n_block_size: tl.constexpr, k_block_size: tl.constexpr,
    # Strides for the matrices
    stride_xm, stride_xk,
    stride_yk, stride_yn,
    stride_zm, stride_zn,
    # Optional accumulator type
    ACC_TYPE: tl.constexpr
):
    # Compute the program ID and corresponding block indices
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(m_size, m_block_size)
    block_m = pid // (num_blocks_m)
    block_n = pid % (num_blocks_m)

    # Compute offsets for the current block
    offs_m = block_m * m_block_size + tl.arange(0, m_block_size)
    offs_n = block_n * n_block_size + tl.arange(0, n_block_size)
    offs_k = tl.arange(0, k_block_size)

    # Create a mask for bounds checking
    x_mask = offs_m[:, None] < m_size
    y_mask = offs_n[None, :] < n_size

    # Initialize accumulator
    acc = tl.zeros((m_block_size, n_block_size), dtype=ACC_TYPE)

    # Iterate over k dimension
    for k in range(0, k_size, k_block_size):
        # Load blocks from x and y matrices
        x_block_ptr = x_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk
        y_block_ptr = y_ptr + (k + offs_k[:, None]) * stride_yk + offs_n[None, :] * stride_yn

        x = tl.load(x_block_ptr, mask=x_mask[:, None] & (k + offs_k[None, :] < k_size))
        y = tl.load(y_block_ptr, mask=(k + offs_k[:, None] < k_size) & y_mask[None, :])

        # Compute matrix multiplication for this block
        acc += tl.dot(x, y)

    # Store the result
    z_block_ptr = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
    tl.store(z_block_ptr, acc, mask=x_mask[:, None] & y_mask[None, :])

# Wrapper function
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Check input dimensions
    assert len(x.shape) == 2 and len(y.shape) == 2, "Input tensors must be 2D"
    assert x.shape[1] == y.shape[0], "Inner dimensions must match"
    
    # Extract dimensions
    m_size, k_size = x.shape
    _, n_size = y.shape

    # Define block sizes (these can be tuned for performance)
    m_block_size = 16
    n_block_size = 16
    k_block_size = 16

    # Compute grid size
    grid = lambda meta: (
        triton.cdiv(m_size, m_block_size) * triton.cdiv(n_size, n_block_size),
    )

    # Create output tensor
    z = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)

    # Launch kernel
    matmul_kernel[grid](
        x, y, z,
        m_size, n_size, k_size,
        m_block_size, n_block_size, k_block_size,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        tl.float32  # Accumulator type
    )

    return z
