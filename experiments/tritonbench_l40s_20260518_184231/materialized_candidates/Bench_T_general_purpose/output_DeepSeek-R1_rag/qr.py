import torch
import triton
import triton.language as tl

@triton.jit
def qr_kernel(
    A_ptr, Q_ptr, R_ptr,
    m, n, k,
    stride_A_batch, stride_A_row, stride_A_col,
    stride_Q_batch, stride_Q_row, stride_Q_col,
    stride_R_batch, stride_R_row, stride_R_col,
    compute_q: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid

    A_batch = A_ptr + batch_idx * stride_A_batch
    Q_batch = Q_ptr + batch_idx * stride_Q_batch if compute_q else 0
    R_batch = R_ptr + batch_idx * stride_R_batch

    for j in range(k):
        col_j = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)
        for row_offset in range(0, m, BLOCK_SIZE):
            rows = row_offset + tl.arange(0, BLOCK_SIZE)
            mask = rows < m
            a = tl.load(A_batch + rows * stride_A_row + j * stride_A_col, mask=mask, other=0.0, eviction_policy="evict_first")
            col_j = tl.where(rows < m, a, col_j)

        for i in range(j):
            q_i = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)
            for row_offset in range(0, m, BLOCK_SIZE):
                rows = row_offset + tl.arange(0, BLOCK_SIZE)
                mask = rows < m
                q = tl.load(Q_batch + rows * stride_Q_row + i * stride_Q_col, mask=mask, other=0.0, eviction_policy="evict_first")
                q_i = tl.where(rows < m, q, q_i)
            if DTYPE in (tl.complex64, tl.complex128):
                q_i_conj = tl.conj(q_i)
            else:
                q_i_conj = q_i
            r_ij = tl.sum(q_i_conj * col_j)
            tl.store(R_batch + i * stride_R_row + j * stride_R_col, r_ij)

            col_j = col_j - r_ij * q_i

        r_jj = tl.sqrt(tl.sum(tl.real(col_j * tl.conj(col_j))) if DTYPE in (tl.complex64, tl.complex128) else tl.sum(col_j * col_j))
        tl.store(R_batch + j * stride_R_row + j * stride_R_col, r_jj)

        if compute_q:
            q_j = col_j / r_jj
            for row_offset in range(0, m, BLOCK_SIZE):
                rows = row_offset + tl.arange(0, BLOCK_SIZE)
                mask = rows < m
                tl.store(Q_batch + rows * stride_Q_row + j * stride_Q_col, q_j, mask=mask)

def qr(A: torch.Tensor, mode: str = 'reduced', *, out=None) -> (torch.Tensor, torch.Tensor):
    assert A.dim() >= 2, "Input must be at least 2D"
    m, n = A.shape[-2], A.shape[-1]
    batch_dims = A.shape[:-2]
    num_batches = A.numel() // (m * n)
    k = min(m, n)

    if mode not in ['reduced', 'complete', 'r']:
        raise ValueError(f"mode must be 'reduced', 'complete', or 'r', but got {mode}")
    compute_q = mode != 'r'

    if mode == 'complete':
        q_shape = list(batch_dims) + [m, m]
        r_shape = list(batch_dims) + [m, n]
    else:
        q_shape = list(batch_dims) + [m, k]
        r_shape = list(batch_dims) + [k, n]

    Q = torch.empty(q_shape, dtype=A.dtype, device=A.device) if compute_q else torch.tensor([], dtype=A.dtype, device=A.device)
    R = torch.empty(r_shape, dtype=A.dtype, device=A.device)

    A_flat = A.reshape(-1, m, n)
    Q_flat = Q.reshape(-1, q_shape[-2], q_shape[-1]) if compute_q else None
    R_flat = R.reshape(-1, *r_shape[-2:])

    BLOCK_SIZE = triton.next_power_of_2(max(m, n))
    DTYPE = tl.float32
    if A.dtype == torch.float64:
        DTYPE = tl.float64
    elif A.dtype == torch.complex64:
        DTYPE = tl.complex64
    elif A.dtype == torch.complex128:
        DTYPE = tl.complex128

    grid = (num_batches,)
    qr_kernel[grid](
        A_flat, Q_flat, R_flat,
        m, n, k,
        A_flat.stride(0), A_flat.stride(1), A_flat.stride(2),
        Q_flat.stride(0) if compute_q else 0, Q_flat.stride(1) if compute_q else 0, Q_flat.stride(2) if compute_q else 0,
        R_flat.stride(0), R_flat.stride(1), R_flat.stride(2),
        compute_q=compute_q,
        DTYPE=DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if out is not None:
        out[0].copy_(Q)
        out[1].copy_(R)
        return out
    return (Q, R)
