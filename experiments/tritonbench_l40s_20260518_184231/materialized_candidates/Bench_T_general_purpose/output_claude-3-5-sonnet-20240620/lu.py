import torch
import triton
import triton.language as tl
import math

def lu(A: torch.Tensor, *, pivot: bool = True, out: tuple = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the LU decomposition with optional pivoting of a matrix A.
    
    Args:
        A (Tensor): tensor of shape (*, m, n) where * represents batch dimensions
        pivot (bool, optional): whether to use partial pivoting. Default: True
        out (tuple, optional): output tuple of (P, L, U). Default: None
    
    Returns:
        tuple[Tensor, Tensor, Tensor]: (P, L, U) where:
            - P is the permutation matrix (empty if pivot=False)
            - L is lower triangular with unit diagonal
            - U is upper triangular
    """
    if not torch.is_tensor(A):
        raise TypeError("Input A must be a tensor")
    
    if A.dim() < 2:
        raise ValueError("Input A must have at least 2 dimensions")
    
    # Get matrix dimensions
    *batch_dims, m, n = A.shape
    device = A.device
    dtype = A.dtype
    
    # Initialize output tensors if not provided
    if out is None:
        if pivot:
            P = torch.eye(m, device=device, dtype=dtype).expand(*batch_dims, m, m).contiguous()
        else:
            P = torch.empty((*batch_dims, 0, 0), device=device, dtype=dtype)
        L = torch.zeros((*batch_dims, m, min(m, n)), device=device, dtype=dtype)
        U = torch.zeros((*batch_dims, min(m, n), n), device=device, dtype=dtype)
    else:
        P, L, U = out
    
    # Handle batch dimensions
    batch_size = math.prod(batch_dims) if batch_dims else 1
    A_2d = A.reshape(-1, m, n)
    
    # Define grid and block sizes for Triton kernel
    BLOCK_SIZE = 32
    grid = (triton.cdiv(m, BLOCK_SIZE), triton.cdiv(n, BLOCK_SIZE), batch_size)
    
    # Launch Triton kernel
    _lu_decomposition_kernel[grid](
        A_2d.data_ptr(),
        P.reshape(-1, m, m).data_ptr() if pivot else 0,
        L.reshape(-1, m, min(m, n)).data_ptr(),
        U.reshape(-1, min(m, n), n).data_ptr(),
        m, n, batch_size,
        A_2d.stride(0), A_2d.stride(1), A_2d.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return P, L, U

@triton.jit
def _lu_decomposition_kernel(
    A_ptr, P_ptr, L_ptr, U_ptr,
    M, N, BATCH,
    A_batch_stride, A_row_stride, A_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for LU decomposition.
    
    Args:
        A_ptr: Pointer to input matrix
        P_ptr: Pointer to permutation matrix (0 if no pivoting)
        L_ptr: Pointer to lower triangular matrix
        U_ptr: Pointer to upper triangular matrix
        M, N: Matrix dimensions
        BATCH: Batch size
        A_batch_stride, A_row_stride, A_col_stride: Strides for input tensor
        BLOCK_SIZE: Size of thread block
    """
    pid = tl.program_id(0)
    bid = tl.program_id(2)  # batch index
    
    # Compute offsets for this block
    row_start = pid * BLOCK_SIZE
    
    # Load block of matrix A
    row = row_start + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    mask = (row[:, None] < M) & (col[None, :] < N)
    
    # Offset for batch
    batch_offset = bid * A_batch_stride
    
    # Load block
    a = tl.load(A_ptr + batch_offset + row[:, None] * A_row_stride + col[None, :] * A_col_stride, mask=mask)
    
    # Perform LU decomposition within block
    for k in range(min(BLOCK_SIZE, M, N)):
        if P_ptr != 0:  # If pivoting is enabled
            # Find pivot
            max_val = tl.abs(a[k, k])
            pivot_row = k
            
            for i in range(k + 1, min(M, row_start + BLOCK_SIZE)):
                if tl.abs(a[i, k]) > max_val:
                    max_val = tl.abs(a[i, k])
                    pivot_row = i
            
            # Swap rows if necessary
            if pivot_row != k:
                for j in range(k, N):
                    temp = a[k, j]
                    a[k, j] = a[pivot_row, j]
                    a[pivot_row, j] = temp
                
                # Update permutation matrix
                if P_ptr != 0:
                    for j in range(M):
                        temp = tl.load(P_ptr + batch_offset + k * M + j)
                        tl.store(P_ptr + batch_offset + k * M + j, 
                                tl.load(P_ptr + batch_offset + pivot_row * M + j))
                        tl.store(P_ptr + batch_offset + pivot_row * M + j, temp)
        
        # Compute multipliers
        if a[k, k] != 0:
            for i in range(k + 1, min(M, row_start + BLOCK_SIZE)):
                multiplier = a[i, k] / a[k, k]
                # Store in L
                tl.store(L_ptr + batch_offset + i * M + k, multiplier)
                # Update row
                for j in range(k, N):
                    a[i, j] = a[i, j] - multiplier * a[k, j]
    
    # Store results in U
    for i in range(min(M, BLOCK_SIZE)):
        for j in range(min(N, BLOCK_SIZE)):
            if i <= j:  # Upper triangular part
                tl.store(U_ptr + batch_offset + i * N + j, a[i, j])
    
    # Set unit diagonal in L
    if row_start == 0:
        for i in range(min(M, N)):
            tl.store(L_ptr + batch_offset + i * M + i, 1.0)
