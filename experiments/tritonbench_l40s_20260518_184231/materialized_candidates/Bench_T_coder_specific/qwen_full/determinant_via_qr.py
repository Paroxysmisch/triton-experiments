import torch
import triton
import triton.language as tl

@triton.jit
def determinant_via_qr_triton(A, mode, out):
    # Triton kernel to compute the determinant using QR decomposition
    batch, n = A.shape
    sgn_q = torch.ones((batch, ), dtype=torch.float32, device=A.device)
    d = torch.ones((batch, ), dtype=torch.float32, device=A.device)
    if mode == 'reduced':
        for k in range(n):
            # Apply Givens rotation to zero out elements below the diagonal
            g = givens(A, k, k + 1, n)
            A = apply_givens(A, g, k, k + 1, n)
            d_k = torch.prod(torch.abs(A[:, k, k]))
            d *= d_k
            if A[:, k, k] < 0:
                sgn_q *= -1
    else:
        raise NotImplementedError
    if A.dtype.is_fp64():
        d = d.to(torch.float64)
    sgn_q = sgn_q.to(A.dtype)
    # Compute the determinant as the product of sgn_q and the diagonal elements of R
    return torch.prod(sgn_q * d.unsqueeze(1))

def determinant_via_qr(A, *, mode='reduced', out=None):
    # Wrapper function for the Triton kernel
    if A.is_floating_point():
        if A.dtype in (torch.float32, torch.float64):
            if A.dim() == 2:
                batch, n = A.shape
                A = A.unsqueeze(0) if batch == 1 else A
                out = torch.empty((batch, ), dtype=A.dtype, device=A.device) if out is None else out
                assert out.shape == (batch, )
                determinant_via_qr_triton(A, mode, out)
                return out.squeeze(0) if batch == 1 else out
            else:
                raise ValueError("only 2D tensors are supported")
        else:
            raise ValueError("input must be a floating-point tensor")
    else:
        raise ValueError("input must be a floating-point tensor")
