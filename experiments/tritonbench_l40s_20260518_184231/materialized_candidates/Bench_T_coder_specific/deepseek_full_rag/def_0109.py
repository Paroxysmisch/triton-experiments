import torch
import triton
import triton.language as tl

@triton.jit
def pseudoinverse_svd_kernel(A, s, U, Vt, m, n, rcond: tl.constexpr, full_matrices: tl.constexpr):
    # Compute the SVD of A
    truncated_svd(A, s, U, Vt, m, n)

    # Invert the singular values
    s_inv = tl.where(tl.arange(0, tl.minimum(m, n)) < tl.minimum(m, n), 1.0 / tl.sqrt(s), 0.0)

    # Compute the pseudoinverse
    pseudoinverse_kernel(A, U, Vt, s_inv, m, n, rcond, full_matrices)

@triton.jit
def pseudoinverse_kernel(A, U, Vt, s_inv, m, n, rcond: tl.constexpr, full_matrices: tl.constexpr):
    # Compute the pseudoinverse
    Vt_s_inv = Vt[:, :tl.minimum(m, n)] * s_inv[None, :]
    U_Vt_s_inv = tl.dot(U[:, :tl.minimum(m, n)], Vt_s_inv)
    A_pinv = U_Vt_s_inv

    # Optionally compute the Moore-Penrose pseudoinverse
    if not full_matrices:
        U_rcond = U[:, tl.minimum(m, n):]
        A_pinv = tl.concatenate([A_pinv, U_rcond * 0.0], axis=1)

    tl.store(A, A_pinv)

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    # Validate input
    assert A.is_contiguous(), "Input tensor must be contiguous"
    assert A.ndim >= 2, "Input tensor must have at least 2 dimensions"
    m, n = A.shape[-2:]
    assert n > 0, "Input tensor must have at least one column"

    # Create output tensor
    if out is None:
        dtype = A.dtype
        if A.dtype in (torch.float32, torch.complex64):
            dtype = torch.get_default_dtype()
        out = torch.empty(m, n, device=A.device, dtype=dtype)
    else:
        assert out.shape == (m, n), "Output tensor must have the same shape as the input tensor"

    # Compute the pseudoinverse
    s = torch.empty(min(m, n), device=A.device, dtype=A.dtype)
    U = torch.empty(m, m, device=A.device, dtype=A.dtype)
    Vt = torch.empty(n, n, device=A.device, dtype=A.dtype)
    pseudoinverse_svd_kernel[(1,)](A, s, U, Vt, m, n, rcond, full_matrices)

    return out
