import torch

def matrix_factorization(A, rank=None, out=None):
    if rank is None:
        rank = min(A.shape[-2:]) - 1
    if rank > 10:
        rank = 10
    U, S, Vh = torch.linalg.svd(A)
    U = U[..., :rank, :]
    S = S[..., :rank]
    Vh = Vh[..., :rank, :]
    if out is not None:
        out.copy_(U @ torch.diag(S) @ Vh.mT)
        return out
    else:
        return U @ torch.diag(S) @ Vh.mT
