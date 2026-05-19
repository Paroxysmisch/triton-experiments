import torch
import triton
import triton.language as tl

@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    b_mask = offset_m < M

    A = A + offset_n * M + offset_m
    B = B + offset_m
    C = C + offset_n

    a = tl.load(A, mask=n_mask & b_mask, other=0.0).to(tl.float32)
    b = tl.load(B, mask=b_mask, other=0.0).to(tl.float32)

    c = tl.dot(a, b)

    tl.store(C, c.to(C.dtype.element_ty), mask=n_mask)


def mv(inp, vec):
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch.cuda.device(inp.device):
        mv_kernel[grid](inp, vec, out, N, M, BLOCK_N=128, BLOCK_M=128)
    return out
