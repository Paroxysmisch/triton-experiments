import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
    ],
    key=['m', 'n'],
)
@triton.jit
def qr_kernel(
    A_ptr, Q_ptr, R_ptr, m, n,
    stride_Ab, stride_Am, stride_An,
    stride_Qb, stride_Qm, stride_Qn,
    stride_Rb, stride_Rm, stride_Rn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_batch = tl.program_id(axis=2)
    pid = tl.program_id(axis=0)
    
    A = tl.make_block_ptr(
        base=A_ptr + pid_batch * stride_Ab,
        shape=(m, n),
        strides=(stride_Am, stride_An),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    Q = tl.make_block_ptr(
        base=Q_ptr + pid_batch * stride_Qb,
        shape=(m, m),
        strides=(stride_Qm, stride_Qn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    R = tl.make_block_ptr(
        base=R_ptr + pid_batch * stride_Rb,
        shape=(m, n),
        strides=(stride_Rm, stride_Rn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    for i in range(0, m):
        for j in range(0, n):
            a = tl.load(A + (i * stride_Am + j * stride_An))
            if i == j:
                tl.store(R + (i * stride_Rm + j * stride_Rn), a)
            else:
                tl.store(R + (i * stride_Rm + j * stride_Rn), 0.0)
            if i < m and j < m:
                tl.store(Q + (i * stride_Qm + j * stride_Qn), 1.0 if i == j else 0.0)

def qr(A: torch.Tensor, mode: str = 'reduced', *, out=None) -> (torch.Tensor, torch.Tensor):
    if mode not in ['reduced', 'complete', 'r']:
        raise ValueError(f"mode must be one of 'reduced', 'complete', 'r', but got {mode}")
    
    if A.numel() == 0:
        raise RuntimeError("qr input tensor is empty")
    
    m, n = A.shape[-2], A.shape[-1]
    batch = A.shape[:-2]
    k = min(m, n)
    
    Q = torch.empty((*batch, m, m if mode == 'complete' else k), dtype=A.dtype, device=A.device)
    R = torch.empty((*batch, k, n), dtype=A.dtype, device=A.device)
    
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']), triton.cdiv(n, meta['BLOCK_SIZE_N']), A.numel() // (m * n))
    
    qr_kernel[grid](
        A, Q, R, m, n,
        A.stride(-3) if A.ndim > 2 else 0,
        A.stride(-2), A.stride(-1),
        Q.stride(-3) if Q.ndim > 2 else 0,
        Q.stride(-2), Q.stride(-1),
        R.stride(-3) if R.ndim > 2 else 0,
        R.stride(-2), R.stride(-1)
    )
    
    if mode == 'reduced':
        Q = Q[..., :k]
    elif mode == 'r':
        Q = torch.empty(0, dtype=A.dtype, device=A.device)
        R = R[..., :k, :]
    
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError("out must be a tuple of two tensors")
        out[0].data = Q
        out[1].data = R
        return out
    else:
        return (Q, R)
