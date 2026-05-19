import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel for computing the Cholesky decomposition
@triton.jit
def cholesky_kernel(A, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx = pid < n

    # Load lower triangular part of A
    a = tl.load(A + idx[:, None] * n + idx[None, :], mask=idx[:, None] & idx[None, :]).to(tl.float32)

    # Compute Cholesky decomposition
    if k == 0:
        l = tl.sqrt(a)
        r = 0.0
    else:
        offset = n * k
        a_p = a + offset
        l = tl.load(A + idx[:, None] * n + offset + idx[None, :], mask=idx[:, None] & idx[None, :]).to(tl.float32)
        a_pp = tl.sum(l * l)
        a_pq = tl.sum(l * a_p)
        q = a_pp.to(tl.float32)
        r = a_pq / q
        l = a_p / q

    # Store results
    offset = n * k
    p = tl.program_id(0)
    idx2 = p * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx3 = idx2 < n + offset
    tl.store(A + idx2[:, None] * n + offset + idx2[None, :], l, mask=idx3[:, None] & idx3[None, :])
    tl.store(A + idx2, r, mask=(tl.arange(0, BLOCK_SIZE) < 1) & idx3)

# Wrapper function for calling the Triton kernel
def cholesky(A, upper=False, out=None):
    A = A.contiguous()
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"
    n = A.size(-1)
    assert A.size(-2) == n, "The last two dimensions of input tensor must be equal"

    if out is not None:
        out = out.contiguous()
        assert out.shape == A.shape, "Output tensor shape must match input tensor shape"
    else:
        out = torch.empty_like(A)

    if not upper:
        A = out.tril()
    else:
        A = out.t().tril().t()

    m = volume(A.shape[:-2])
    grid_fn = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE"]))

    with torch.cuda.device(A.device):
        cholesky_kernel[grid_fn](A, n, m, BLOCK_SIZE=32)

    return out
