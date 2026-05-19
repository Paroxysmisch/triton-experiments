import torch
import triton
import triton.language as tl
from triton.ops.eig import eigh
from triton.ops.norm import _make_vector

@triton.jit
def _max_abs_eigenvalue(A):
    # Compute the eigenvalues of A.
    lambda_ = eigh(A)
    # Return the maximum absolute eigenvalue.
    return tl.max(tl.abs(lambda_))

def spectral_norm_eig(A, *, out=None):
    if out is None:
        out = torch.empty(
            (*A.shape[:-2], 1), dtype=A.dtype, device=A.device
        )  # (*, 1)
    else:
        assert out.is_contiguous()
        assert out.shape[-1] == 1
        assert out.dtype == A.dtype
        assert out.device == A.device
    # Compute the spectral norm of A.
    _spectral_norm_eig(A, out)
    return out.view(*A.shape[:-2], 1)  # (*, 1)

@triton.jit
def _spectral_norm_eig(A, out):
    # Compute the spectral norm of A.
    out.data[:] = _max_abs_eigenvalue(A)
