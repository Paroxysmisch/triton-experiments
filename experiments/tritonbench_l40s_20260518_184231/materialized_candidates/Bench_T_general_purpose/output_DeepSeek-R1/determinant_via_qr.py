import torch
import triton
import triton.language as tl

@triton.jit
def diag_prod_kernel(
    r_ptr,
    prod_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n:
        return
    row = pid
    col = pid
    stride = r_ptr.stride(0)
    offset = row * stride + col
    element = tl.load(r_ptr + offset)
    tl.atomic_mul(prod_ptr, element)

def diag_prod_triton(R):
    n = R.size(-1)
    prod = torch.ones((), dtype=R.dtype, device=R.device)
    grid = lambda meta: (n,)
    diag_prod_kernel[grid](R, prod, n, BLOCK_SIZE=1)
    return prod

def determinant_via_qr(A, *, mode='reduced', out=None):
    # Perform QR decomposition
    Q, R = torch.linalg.qr(A, mode=mode)
    
    # Compute determinant of Q using PyTorch's determinant function
    det_Q = torch.det(Q)
    
    # Compute product of the diagonal elements of R using Triton kernel
    product_R = diag_prod_triton(R)
    
    # Compute the final determinant
    det_A = det_Q * product_R
    
    # Handle output tensor if provided
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a Tensor")
        out.copy_(det_A)
        return out
    return det_A
