import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

# Triton kernel for computing the determinant of a square matrix
@triton.jit
def det(A, stride_za, out, xnumel, **meta):
    TN = meta["TN"]
    BLOCK_SIZE = meta["BLOCK_SIZE"]
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < xnumel
    # reshape A
    Az = tl.make_block_ptr(A, (TN, TN), (1, TN), (0, 1), (0, 0), (TN, 1), (1, 0))
    # permute A
    A = tl.load(Az, boundary_check=(0, 1), padding_option="zero", mask=mask)
    det = tl.linalg.det(A)
    # store det
    tl.store(out + offs, det, mask=mask)

# Wrapper function for computing the determinant of a square matrix
def det_wrapper(A: Tensor, *, out: Tensor = None):
    if not A.is_floating_point():
        raise ValueError("expected a floating point tensor")
    if A.ndim < 2:
        raise ValueError("expected a tensor with at least 2 dimensions")
    n = A.shape[-2]
    if n != A.shape[-1]:
        raise ValueError("expected a square matrix")
    if out is None:
        out = torch.empty(A.shape[:-2] + (1,), dtype=A.dtype, device=A.device)
    else:
        if out.ndim < 1:
            raise ValueError("expected a tensor with at least 1 dimension")
        if out.shape[:-1] != A.shape[:-2]:
            raise ValueError("shape mismatch")
        if not out.is_floating_point():
            raise ValueError("expected a floating point tensor for out")
    xnumel = out.numel()
    with torch.cuda.device(A.device):
        grid = lambda meta: (triton.cdiv(xnumel, meta["BLOCK_SIZE"]),)
        det[grid](A, A.stride(-2), out, xnumel, TN=n, BLOCK_SIZE=1024)
    return out
