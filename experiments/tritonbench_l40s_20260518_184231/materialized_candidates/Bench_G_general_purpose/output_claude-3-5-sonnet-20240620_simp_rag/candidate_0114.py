import triton
import triton.language as tl
import torch

@triton.jit
def kernel(
    M_ptr, Out_ptr,
    stride_m_row, stride_m_col,  # Strides for input matrix M
    stride_out_row, stride_out_col,  # Strides for output matrix Out
    SIZE_M, D_HEAD,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate block indices
    num_blocks_m = tl.cdiv(SIZE_M, BLOCK_SIZE_M)
    block_m = pid // num_blocks_m
    block_d = pid % num_blocks_m

    # Calculate offsets
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = block_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    
    # Create masks to handle boundary conditions
    mask_m = offs_m < SIZE_M
    mask_d = offs_d < D_HEAD
    
    # Compute memory locations for load and store
    m_ptrs = M_ptr + offs_m[:, None] * stride_m_row + offs_d[None, :] * stride_m_col
    out_ptrs = Out_ptr + offs_d[:, None] * stride_out_row + offs_m[None, :] * stride_out_col
    
    # Load input matrix block with masking
    x = tl.load(m_ptrs, mask=mask_m[:, None] & mask_d[None, :])
    
    # Transpose the block
    x_trans = tl.trans(x)
    
    # Store transposed block with masking
    tl.store(out_ptrs, x_trans, mask=mask_d[:, None] & mask_m[None, :])

def transpose_matrix(M: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to transpose a matrix using Triton kernel
    
    Args:
        M: Input matrix of shape (SIZE_M, D_HEAD)
    
    Returns:
        Transposed matrix of shape (D_HEAD, SIZE_M)
    """
    # Get matrix dimensions
    SIZE_M, D_HEAD = M.shape
    
    # Initialize output tensor
    Out = torch.empty((D_HEAD, SIZE_M), device=M.device, dtype=M.dtype)
    
    # Define block sizes (can be tuned for better performance)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_D = 32
    
    # Calculate grid size
    grid = (triton.cdiv(SIZE_M, BLOCK_SIZE_M) * triton.cdiv(D_HEAD, BLOCK_SIZE_D),)
    
    # Launch kernel
    kernel[grid](
        M, Out,
        M.stride(0), M.stride(1),
        Out.stride(0), Out.stride(1),
        SIZE_M, D_HEAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )
    
    return Out
