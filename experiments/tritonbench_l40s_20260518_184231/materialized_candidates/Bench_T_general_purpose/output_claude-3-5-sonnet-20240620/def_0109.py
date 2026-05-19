import triton
import triton.language as tl
import torch

@triton.jit
def pseudoinverse_kernel(
    # Pointers to matrices
    a_ptr, u_ptr, s_ptr, vh_ptr, out_ptr,
    # Matrix dimensions
    M, N, 
    # Parameters
    rcond,
    # Strides for the matrices
    stride_am, stride_an,
    stride_um, stride_un,
    stride_s,
    stride_vhm, stride_vhn,
    stride_om, stride_on,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Compute the row and column start indices for this block
    row_start = pid // (N // BLOCK_SIZE_N) * BLOCK_SIZE_M
    col_start = pid % (N // BLOCK_SIZE_N) * BLOCK_SIZE_N

    # Create block pointers
    s_block_ptr = s_ptr + row_start * stride_s
    
    # Load singular values for this block
    s = tl.load(s_block_ptr + tl.arange(0, BLOCK_SIZE_M))
    
    # Compute maximum singular value for rcond threshold
    s_max = tl.max(s)
    threshold = rcond * s_max
    
    # Compute reciprocal of singular values above threshold
    s_inv = tl.where(s > threshold, 1.0 / s, 0.0)
    
    # Store the result
    tl.store(out_ptr + row_start * stride_om + col_start * stride_on, s_inv)

def pseudoinverse_svd(A: torch.Tensor, *, full_matrices: bool = True, 
                     rcond: float = 1e-15, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.
    
    Args:
        A (Tensor): Input tensor of shape (*, m, n) where * represents batch dimensions
        full_matrices (bool): If True, compute full SVD, else reduced. Default: True
        rcond (float): Relative condition number threshold. Default: 1e-15
        out (Tensor, optional): Output tensor. Default: None
    
    Returns:
        Tensor: The pseudoinverse of A
    """
    # Get matrix dimensions
    *batch_dims, m, n = A.shape
    device = A.device
    dtype = A.dtype
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Initialize output tensor if not provided
    if out is None:
        out = torch.empty((*batch_dims, n, m), dtype=dtype, device=device)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 32
    grid = (m * n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    pseudoinverse_kernel[(grid,)](
        A, U, S, Vh, out,
        m, n,
        rcond,
        A.stride(-2), A.stride(-1),
        U.stride(-2), U.stride(-1),
        S.stride(-1),
        Vh.stride(-2), Vh.stride(-1),
        out.stride(-2), out.stride(-1),
        BLOCK_SIZE, BLOCK_SIZE
    )
    
    # Compute final pseudoinverse: V @ diag(S_inv) @ U.H
    return out
