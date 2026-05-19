import triton
import triton.language as tl
import torch

@triton.jit
def lu_decomposition_kernel(
    A_ptr, L_ptr, U_ptr, P_ptr, 
    stride_am, stride_an,
    stride_lm, stride_ln,
    stride_um, stride_un,
    stride_pm,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute row and col indices
    row = pid // (N // BLOCK_SIZE)
    col = pid % (N // BLOCK_SIZE)
    
    # Load block
    offs_am = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_an = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_am[:, None] < M) & (offs_an[None, :] < N)
    
    # Initialize L, U, and P
    a = tl.load(A_ptr + offs_am[:, None] * stride_am + offs_an[None, :] * stride_an, mask=mask)
    l = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    u = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    p = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Perform block LU decomposition
    for k in range(BLOCK_SIZE):
        # Find pivot
        if row == 0:
            max_val = tl.abs(a[k, k])
            max_idx = k
            for i in range(k + 1, BLOCK_SIZE):
                if tl.abs(a[i, k]) > max_val:
                    max_val = tl.abs(a[i, k])
                    max_idx = i
            
            # Swap rows if necessary
            if max_idx != k:
                temp = a[k, :].copy()
                a[k, :] = a[max_idx, :]
                a[max_idx, :] = temp
                p[k] = max_idx
        
        # Update L and U blocks
        if mask[k, k]:
            l[k, k] = 1.0
            u[k, k] = a[k, k]
            
            # Update L
            for i in range(k + 1, BLOCK_SIZE):
                if mask[i, k]:
                    l[i, k] = a[i, k] / u[k, k]
            
            # Update U
            for j in range(k + 1, BLOCK_SIZE):
                if mask[k, j]:
                    u[k, j] = a[k, j]
            
            # Update A
            for i in range(k + 1, BLOCK_SIZE):
                for j in range(k + 1, BLOCK_SIZE):
                    if mask[i, j]:
                        a[i, j] = a[i, j] - l[i, k] * u[k, j]
    
    # Store results
    tl.store(L_ptr + offs_am[:, None] * stride_lm + offs_an[None, :] * stride_ln, l, mask=mask)
    tl.store(U_ptr + offs_am[:, None] * stride_um + offs_an[None, :] * stride_un, u, mask=mask)
    if row == 0:
        tl.store(P_ptr + offs_am * stride_pm, p)

def invert_matrix_lu(A: torch.Tensor, *, pivot: bool = True, out: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the inverse of a square matrix using LU decomposition.
    
    Args:
        A: Input square matrix or batch of matrices
        pivot: Whether to use partial pivoting (default: True)
        out: Optional output tensor
    
    Returns:
        Inverse of input matrix A
    """
    if A.dim() < 2:
        raise ValueError("Input tensor must be at least 2D")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input must be a square matrix")
    
    # Handle batch dimensions
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(A)
    elif out.shape != A.shape:
        raise ValueError("Output tensor must have the same shape as input")
    
    # Allocate temporary tensors
    device = A.device
    dtype = A.dtype
    L = torch.empty_like(A)
    U = torch.empty_like(A)
    P = torch.empty(*batch_dims, n, dtype=torch.int32, device=device)
    
    # Configure kernel parameters
    BLOCK_SIZE = 32
    grid = (n * n) // (BLOCK_SIZE * BLOCK_SIZE)
    
    # Launch LU decomposition kernel
    lu_decomposition_kernel[(grid,)](
        A_ptr=A, L_ptr=L, U_ptr=U, P_ptr=P,
        stride_am=A.stride(-2), stride_an=A.stride(-1),
        stride_lm=L.stride(-2), stride_ln=L.stride(-1),
        stride_um=U.stride(-2), stride_un=U.stride(-1),
        stride_pm=P.stride(-1),
        M=n, N=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Solve L Y = P for Y
    Y = torch.triangular_solve(P, L, upper=False)[0]
    
    # Solve U X = Y for X (the inverse)
    out = torch.triangular_solve(Y, U, upper=True)[0]
    
    return out
