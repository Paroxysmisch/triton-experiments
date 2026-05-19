import torch

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    # Compute the SVD of A
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Truncate the singular values and vectors to rank k
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    # Efficiently compute Ak = U_k * diag(S_k) @ Vh_k by scaling U_k's columns with S_k and then multiplying with Vh_k
    Ak = (U_k * S_k.unsqueeze(-2)) @ Vh_k
    
    # Handle the output tensor if provided
    if out is not None:
        out.copy_(Ak)
        return out
    return Ak
