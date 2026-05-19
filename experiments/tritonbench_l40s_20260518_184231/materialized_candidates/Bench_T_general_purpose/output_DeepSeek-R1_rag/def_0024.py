import torch
import triton
import triton.language as tl

@triton.jit
def batched_triangular_solve_kernel(
    R_ptr,
    B_ptr,
    X_ptr,
    stride_R_batch,
    stride_R_row,
    stride_R_col,
    stride_B_batch,
    stride_B_row,
    stride_B_col,
    stride_X_batch,
    stride_X_row,
    stride_X_col,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = pid // k
    col_id = pid % k

    R_batch_offset = batch_id * stride_R_batch
    B_batch_offset = batch_id * stride_B_batch
    X_batch_offset = batch_id * stride_X_batch

    B_col_offset = col_id * stride_B_col
    X_col_offset = col_id * stride_X_col

    x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(n-1, -1, -1):
        r_ii_ptr = R_ptr + R_batch_offset + i * stride_R_row + i * stride_R_col
        r_ii = tl.load(r_ii_ptr)
        sum_val = 0.0
        for j in range(i+1, n):
            r_ij_ptr = R_ptr + R_batch_offset + i * stride_R_row + j * stride_R_col
            r_ij = tl.load(r_ij_ptr)
            sum_val += r_ij * x[j]
        b_i_ptr = B_ptr + B_batch_offset + i * stride_B_row + B_col_offset
        b_i = tl.load(b_i_ptr)
        x_i = (b_i - sum_val) / r_ii
        x = tl.store(x, i, x_i)

    for i in range(n):
        x_i_ptr = X_ptr + X_batch_offset + i * stride_X_row + X_col_offset
        tl.store(x_i_ptr, x[i])

def least_squares_qr(A: torch.Tensor, b: torch.Tensor, *, mode='reduced', out=None) -> torch.Tensor:
    assert A.shape[:-2] == b.shape[:-1], "Batch dimensions of A and b must match"
    m, n = A.shape[-2], A.shape[-1]
    Q, R = torch.linalg.qr(A, mode=mode)
    
    if mode == 'complete':
        Q = Q[..., :n]
        R = R[..., :n, :n]
    
    Q_H = Q.mT if Q.is_complex() else Q.transpose(-2, -1)
    B = torch.matmul(Q_H, b.unsqueeze(-1) if b.ndim == Q.ndim -1 else b)
    
    if B.ndim < R.ndim:
        B = B.unsqueeze(-1)
    
    batch_shape = R.shape[:-2]
    num_batches = torch.tensor(batch_shape).prod().item() if batch_shape else 1
    k = B.shape[-1]
    R_expanded = R.contiguous().view(-1, n, n)
    B_expanded = B.contiguous().view(-1, n, k)
    X = torch.empty_like(B_expanded)
    
    grid = lambda _: (num_batches * k,)
    BLOCK_SIZE = triton.next_power_of_2(n)
    
    batched_triangular_solve_kernel[grid](
        R_expanded, B_expanded, X,
        R_expanded.stride(0), R_expanded.stride(1), R_expanded.stride(2),
        B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2),
        X.stride(0), X.stride(1), X.stride(2),
        n, k, BLOCK_SIZE=BLOCK_SIZE
    )
    
    X = X.view(*batch_shape, n, k).squeeze(-1) if k == 1 else X.view(*batch_shape, n, k)
    
    if out is not None:
        out.copy_(X)
        return out
    return X
