import triton
import numpy as np

def spectral_norm_eig(A, out=None):
    # Compute eigenvalues
    eigvals = triton.linalg.eigvals(A)
    # Compute absolute values
    eigvals_abs = triton.abs(eigvals)
    # Compute maximum absolute eigenvalue
    max_eigval_abs = triton.max(eigvals_abs, axis=-1, keepdims=True)
    # If an output tensor is provided, copy the result to it
    if out is not None:
        triton.copy(max_eigval_abs, out)
    return max_eigval_abs
