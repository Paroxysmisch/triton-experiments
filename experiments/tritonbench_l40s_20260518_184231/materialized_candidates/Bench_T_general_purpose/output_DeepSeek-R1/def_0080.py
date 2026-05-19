import torch
import triton
import triton.language as tl

@triton.jit
def tri_solve_kernel(
    R_ptr, y_ptr, x_ptr,
    n, k,
    stride_R_row, stride_R_col,
    stride_y_row, stride_y_col,
    stride_x_row, stride_x_col,
    BLOCK_SIZE: tl.constexpr,
):
    col_idx = tl.program_id(0)
    if col_idx >= k:
        return

    # Pointers to the current column in y and x
    y_col_ptr = y_ptr + col_idx * stride_y_col
    x_col_ptr = x_ptr + col_idx * stride_x_col

    # Temporary array to store x_col
    x_col = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate from the last row to the first
    for i in range(n-1, -1, -1):
        # Load y[i, col_idx]
        y_i = tl.load(y_col_ptr + i * stride_y_row)
        sum_val = 0.0
        # Compute sum(R[i, j] * x_col[j] for j > i)
        for j in range(i + 1, n):
            R_ij = tl.load(R_ptr + i * stride_R_row + j * stride_R_col)
            x_j = x_col[j]
            sum_val += R_ij * x_j
        # Load R[i, i]
        R_ii = tl.load(R_ptr + i * stride_R_row + i * stride_R_col)
        # Compute x_col[i]
        x_i = (y_i - sum_val) / R_ii
        # Update x_col
        x_col = tl.store(x_col, x_i, [i])

    # Write x_col to global memory
    for i in range(n):
        tl.store(x_col_ptr + i * stride_x_row, x_col[i])

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert A.shape[0] >= A.shape[1], "m >= n required for QR decomposition"
    m, n = A.shape
    k_cols = b.shape[1]

    # Compute QR decomposition
    Q, R = torch.linalg.qr(A, mode='reduced')
    # Compute Q^T @ b
    Q_T_b = torch.matmul(Q.T, b)

    # Allocate output tensor
    x = torch.empty((n, k_cols), device=A.device, dtype=A.dtype)

    # Launch the Triton kernel
    grid = (k_cols,)
    # Ensure 2D tensors are properly strided
    if R.stride(0) < R.stride(1):
        R = R.contiguous()
    if Q_T_b.stride(0) < Q_T_b.stride(1):
        Q_T_b = Q_T_b.contiguous()
    if x.stride(0) < x.stride(1):
        x = x.contiguous()

    BLOCK_SIZE = triton.next_power_of_2(n)
    tri_solve_kernel[grid](
        R, Q_T_b, x,
        n, k_cols,
        R.stride(0), R.stride(1),
        Q_T_b.stride(0), Q_T_b.stride(1),
        x.stride(0), x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return x
