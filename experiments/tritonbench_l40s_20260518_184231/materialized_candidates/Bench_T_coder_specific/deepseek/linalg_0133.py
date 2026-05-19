import torch

def linalg_ldl_factor(A, hermitian=False, out=None):
    # Check if A is on a CUDA device and synchronize
    if A.is_cuda:
        torch.cuda.synchronize()

    # Perform the LDL factorization
    LD, pivots = torch.linalg.ldl_factor(A, hermitian=hermitian, out=out)

    return LD, pivots
