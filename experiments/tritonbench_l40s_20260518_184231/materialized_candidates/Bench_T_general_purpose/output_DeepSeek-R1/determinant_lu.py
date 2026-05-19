import torch
import triton
import triton.language as tl

@triton.jit
def lu_no_pivot_kernel(
    A_ptr, n,
    stride_batch, stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    for k in range(n):
        a_kk_offset = pid_batch * stride_batch + k * stride_row + k * stride_col
        a_kk = tl.load(A_ptr + a_kk_offset)
        for i in range(k + 1, n):
            a_ik_offset = pid_batch * stride_batch + i * stride_row + k * stride_col
            a_ik = tl.load(A_ptr + a_ik_offset)
            factor = a_ik / a_kk
            tl.store(A_ptr + a_ik_offset, factor)
            for j in range(k + 1, n):
                a_ij_offset = pid_batch * stride_batch + i * stride_row + j * stride_col
                a_kj_offset = pid_batch * stride_batch + k * stride_row + j * stride_col
                a_ij = tl.load(A_ptr + a_ij_offset)
                a_kj = tl.load(A_ptr + a_kj_offset)
                a_ij_updated = a_ij - factor * a_kj
                tl.store(A_ptr + a_ij_offset, a_ij_updated)

@triton.jit
def compute_determinant_kernel(
    LU_ptr, pivots_ptr, det_ptr,
    n, pivot,
    stride_LU_batch, stride_LU_row, stride_LU_col,
    stride_pivots_batch, stride_pivots_k,
    stride_det_batch,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_LU = pid * stride_LU_batch
    product = 1.0
    for k in range(n):
        diag_offset = offset_LU + k * (stride_LU_row + stride_LU_col)
        diag_val = tl.load(LU_ptr + diag_offset)
        product = product * diag_val
    sign = 1.0
    if pivot:
        swaps = 0
        offset_pivots = pid * stride_pivots_batch
        for k in range(n - 1):
            pivot_k = tl.load(pivots_ptr + offset_pivots + k * stride_pivots_k)
            if pivot_k != (k + 1):
                swaps += 1
        sign = tl.where(swaps % 2 == 0, 1.0, -1.0)
    det = product * sign
    tl.store(det_ptr + pid * stride_det_batch, det)

def determinant_lu(A: torch.Tensor, *, pivot: bool = True, out: torch.Tensor = None) -> torch.Tensor:
    assert A.dim() >= 2 and A.shape[-1] == A.shape[-2], "A must be a square matrix"
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    num_batches = A.numel() // (n * n)
    A_flat = A.view(-1, n, n)
    if pivot:
        LU, pivots = torch.linalg.lu_factor(A)
        pivots_flat = pivots.view(-1, n-1)
    else:
        LU = A_flat.clone()
        grid = (num_batches,)
        lu_no_pivot_kernel[grid](
            LU, n,
            LU.stride(0), LU.stride(1), LU.stride(2),
            BLOCK_SIZE=min(n, 32)
        )
        pivots_flat = None
    det_flat = torch.empty(num_batches, dtype=A.dtype, device=A.device)
    compute_determinant_kernel[(num_batches,)](
        LU, pivots_flat if pivot else None, det_flat,
        n, pivot,
        LU.stride(0), LU.stride(1), LU.stride(2),
        pivots_flat.stride(0) if pivot else 0, pivots_flat.stride(1) if pivot else 0,
        det_flat.stride(0),
        BLOCK_SIZE=min(n, 32)
    )
    det = det_flat.view(*batch_shape)
    if out is not None:
        out.copy_(det)
    return det
