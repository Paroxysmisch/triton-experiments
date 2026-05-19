import torch
import triton
import triton.language as tl

@triton.jit
def _fused_svd_reconstruct(
    A,
    S,
    U,
    Vh,
    stride_za,
    stride_sa,
    stride_ua,
    stride_va,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_i = i < M

    a = tl.load(A + i * stride_za, mask=mask_i, other=0.0)
    s = tl.load(S + i * stride_sa, mask=mask_i, other=0.0)
    u = tl.load(U + i * stride_ua, mask=mask_i, other=0.0)
    v = tl.load(Vh + i * stride_va, mask=mask_i, other=0.0)

    o = tl.where(mask_i, u * (a * s), 0.0)

    tl.store(A + i * stride_za, o, mask=mask_i)


def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    A = A.contiguous()

    U, S, Vh = torch.svd(A, full_matrices=False)

    K = min(U.size(1), Vh.size(1))
    M, N = U.size(0), Vh.size(0)

    out = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    block_size = 32

    if K >= 4096:
        block_size = 64
    if K >= 8192:
        block_size = 128

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE"]),)
    _fused_svd_reconstruct[grid](
        A,
        S,
        U,
        Vh,
        A.stride(0),
        S.stride(0),
        U.stride(0),
        Vh.stride(0),
        M,
        N,
        K,
        BLOCK_SIZE=block_size,
    )

    return out
