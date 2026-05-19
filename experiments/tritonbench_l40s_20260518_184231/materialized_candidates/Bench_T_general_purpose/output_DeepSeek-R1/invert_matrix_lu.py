import torch

def invert_matrix_lu(A, *, pivot=True, out=None):
    # Check if A is a square matrix or a batch of square matrices
    assert A.shape[-1] == A.shape[-2], "Input must be a square matrix or a batch of square matrices."
    
    # Perform LU decomposition
    LU, pivots = torch.lu(A, pivot=pivot)
    # Unpack into permutation matrix P, lower triangular L, upper triangular U
    P, L, U = torch.lu_unpack(LU, pivots)
    
    # Compute Y = L^{-1} P by solving L Y = P
    Y = torch.linalg.solve_triangular(L, P, upper=False, unitriangular=True)
    
    # Compute A^{-1} = U^{-1} Y by solving U A_inv = Y
    A_inv = torch.linalg.solve_triangular(U, Y, upper=True, unitriangular=False)
    
    # Handle the output tensor if provided
    if out is not None:
        if out.dtype != A_inv.dtype or out.shape != A_inv.shape:
            raise ValueError("Output tensor has incorrect dtype or shape.")
        out.copy_(A_inv)
        return out
    else:
        return A_inv
