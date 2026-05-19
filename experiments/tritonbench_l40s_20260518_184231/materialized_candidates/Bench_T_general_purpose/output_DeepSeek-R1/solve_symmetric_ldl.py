import torch

def solve_symmetric_ldl(A, b, *, hermitian=False, out=None):
    # Compute LDL decomposition
    A_ldl, pivots = torch.linalg.ldl_factor(A, hermitian=hermitian)
    L, D = torch.linalg.ldl_unpack(A_ldl, pivots)
    
    # Reconstruct A as L @ D @ L^T or L @ D @ L^H based on the hermitian flag
    if hermitian:
        A_reconstructed = L @ (D @ L.mT.conj())
    else:
        A_reconstructed = L @ (D @ L.mT)
    
    # Solve the linear system using the reconstructed matrix
    return torch.linalg.solve(A_reconstructed, b, out=out)
