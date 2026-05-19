import torch
import triton
import triton.language as tl


@triton.jit
def _solve(A, B, left=False):
    # Get strides
    stride_ama = A.stride(0)
    stride_ana = A.stride(1)
    stride_bmb = B.stride(0)
    stride_bnb = B.stride(1)
    # Get sizes
    batch_shape = broadcast_shapes(list(A.shape)[:-2], list(B.shape)[:-2])
    m = A.shape[-2]
    n = A.shape[-1]
    n_b = B.shape[-1]
    # reshape input data into contiguous 2d tensors
    A_2d = ascontiguous_tensor_view(A, (prod(batch_shape), m, n), (stride_ama, stride_ana, 1))
    B_2d = ascontiguous_tensor_view(B, (prod(batch_shape), m, n_b), (stride_bmb, stride_bnb, 1))
    if left:
        # Compute LU decomposition
        lu_pivots, lu_buffer = torch.lu_factor(A_2d, out_tensor=torch.empty((prod(batch_shape), m, m), dtype=A.dtype, device=A.device))
        # Backsolve using LU decomposition
        sol = torch.lu_solve(B_2d, lu_buffer, lu_pivots)
    else:
        # Reshape input data
        A_t = A_2d.transpose(-1, -2)
        B_t = B_2d.transpose(-1, -2)
        # Compute LU decomposition
        lu_pivots, lu_buffer = torch.lu_factor(A_t, out_tensor=torch.empty((prod(batch_shape), n, n), dtype=A.dtype, device=A.device))
        # Backsolve using LU decomposition
        sol = torch.lu_solve(B_t, lu_buffer, lu_pivots)
        # Reshape output
        sol = sol.transpose(-1, -2)
    # Return to original shape
    out = ascontiguous_tensor_view(sol, list(batch_shape) + [m, n_b], (0,) * len(batch_shape) + (prod(batch_shape) * m, 1))
    return out


def solve(A, B, *, left=False, out=None):
    assert len(A.shape) >= 2, "Matrix A must have at least 2 dimensions"
    assert len(B.shape) >= 2, "Matrix B must have at least 2 dimensions"
    assert A.shape[-2:] == B.shape[-2:], "The last 2 dimensions of both arguments must match"
    assert A.is_contiguous(), "Matrix A must be contiguous"
    if out is not None:
        assert out.shape == broadcast_shapes(A.shape[:-2], B.shape[:-2]) + A.shape[-1:] + B.shape[-1:]
        assert out.is_contiguous()
        return _solve(A, out, left)
    else:
        return _solve(A, B, left)
