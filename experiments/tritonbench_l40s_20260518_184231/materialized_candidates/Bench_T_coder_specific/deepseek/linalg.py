import triton
import torch

def linalg_det(A, *, out=None):
    # Check if A is a batched tensor
    if A.ndim > 2:
        # Compute the determinant for each matrix in the batch
        result = torch.zeros_like(A)
        for i in range(A.shape[0]):
            result[i] = torch.linalg.det(A[i])
    else:
        # Compute the determinant of a single matrix
        result = torch.linalg.det(A)

    # If an output tensor is provided, use it
    if out is not None:
        out.copy_(result)
        return out
    else:
        return result
