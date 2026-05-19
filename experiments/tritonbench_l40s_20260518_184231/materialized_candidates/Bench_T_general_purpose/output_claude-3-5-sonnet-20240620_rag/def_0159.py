import triton
import triton.language as tl
import torch
import math

@triton.jit
def cholesky_kernel(
    A_ptr,          # Pointer to input matrix
    L_ptr,          # Pointer to output matrix
    stride,         # Matrix stride
    n,              # Matrix dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for tiling
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate current block indices
    j = pid % n
    i = pid // n
    
    # Only process elements where i >= j for lower triangular
    if i >= j:
        # Load diagonal element
        a_jj = tl.load(A_ptr + j * stride + j)
        
        if j == 0:
            # First diagonal element is square root
            l_jj = tl.sqrt(a_jj)
            tl.store(L_ptr + j * stride + j, l_jj)
        else:
            # Load row elements for current position
            row_idx = tl.arange(0, j)
            l_row = tl.load(L_ptr + j * stride + row_idx)
            
            # Compute diagonal element
            l_jj = tl.sqrt(a_jj - tl.sum(l_row * l_row))
            tl.store(L_ptr + j * stride + j, l_jj)
            
        # Compute off-diagonal elements
        if i > j:
            # Load necessary elements for computation
            a_ij = tl.load(A_ptr + i * stride + j)
            row_idx = tl.arange(0, j)
            l_i_row = tl.load(L_ptr + i * stride + row_idx)
            l_j_row = tl.load(L_ptr + j * stride + row_idx)
            
            # Compute L[i,j]
            l_ij = (a_ij - tl.sum(l_i_row * l_j_row)) / l_jj
            tl.store(L_ptr + i * stride + j, l_ij)

def cholesky(A: torch.Tensor, *, upper: bool = False, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.
    
    Args:
        A (Tensor): tensor of shape (*, n, n) where * is zero or more batch dimensions
        upper (bool, optional): whether to return an upper triangular matrix
        out (Tensor, optional): output tensor
    
    Returns:
        Tensor: the Cholesky decomposition matrix
    """
    assert A.is_cuda, "Input tensor must be on GPU"
    assert len(A.shape) >= 2, "Input tensor must have at least 2 dimensions"
    assert A.shape[-2] == A.shape[-1], "Input must be a square matrix"
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(A)
    
    # Get matrix dimension
    n = A.shape[-1]
    
    # Process each matrix in the batch
    batch_dims = A.shape[:-2]
    A_flat = A.reshape(-1, n, n)
    out_flat = out.reshape(-1, n, n)
    
    for batch_idx in range(A_flat.shape[0]):
        # Get current matrix
        A_current = A_flat[batch_idx]
        out_current = out_flat[batch_idx]
        
        # Launch kernel
        grid = (n * n,)
        cholesky_kernel[grid](
            A_current.data_ptr(),
            out_current.data_ptr(),
            n,
            n,
            BLOCK_SIZE=32
        )
        
        # Handle upper triangular if requested
        if upper:
            out_current.transpose_(-2, -1).conj_()
    
    # Reshape output back to original dimensions
    out = out_flat.reshape(*batch_dims, n, n)
    
    return out
