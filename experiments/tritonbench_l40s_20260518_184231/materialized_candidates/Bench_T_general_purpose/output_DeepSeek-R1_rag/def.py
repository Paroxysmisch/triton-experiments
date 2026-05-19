import triton
import triton.language as tl
import torch

@triton.jit
def permute_rhs_kernel(
    B_ptr, Y_ptr, pivots_ptr,
    n, k,
    stride_B_batch, stride_B_rhs, stride_B_row,
    stride_Y_batch, stride_Y_rhs, stride_Y_row,
    stride_pivots_batch,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    
    batch_idx = pid0
    rhs_idx = pid1
    row_idx = pid2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < n
    
    # Load pivot indices
    pivot = tl.load(pivots_ptr + batch_idx * stride_pivots_batch + row_idx, mask=mask, other=0)
    
    # Load original B values
    b_val = tl.load(B_ptr + batch_idx * stride_B_batch + rhs_idx * stride_B_rhs + pivot, mask=mask, other=0.0)
    
    # Store permuted values
    tl.store(Y_ptr + batch_idx * stride_Y_batch + rhs_idx * stride_Y_rhs + row_idx, b_val, mask=mask)

@triton.jit
def forward_substitution_kernel(
    L_ptr, Y_ptr,
    n, k,
    stride_L_batch, stride_L_row, stride_L_col,
    stride_Y_batch, stride_Y_rhs, stride_Y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    
    for i in range(n):
        sum_val = 0.0
        for j in range(0, i, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < i
            l = tl.load(L_ptr + pid_batch * stride_L_batch + i * stride_L_row + cols * stride_L_col, mask=mask, other=0.0)
            y = tl.load(Y_ptr + pid_batch * stride_Y_batch + pid_rhs * stride_Y_rhs + cols * stride_Y_row, mask=mask, other=0.0)
            sum_val += tl.sum(l * y)
        y_i = tl.load(Y_ptr + pid_batch * stride_Y_batch + pid_rhs * stride_Y_rhs + i * stride_Y_row)
        y_i -= sum_val
        tl.store(Y_ptr + pid_batch * stride_Y_batch + pid_rhs * stride_Y_rhs + i * stride_Y_row, y_i)

@triton.jit
def backward_substitution_kernel(
    U_ptr, Y_ptr,
    n, k,
    stride_U_batch, stride_U_row, stride_U_col,
    stride_Y_batch, stride_Y_rhs, stride_Y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < n
            u = tl.load(U_ptr + pid_batch * stride_U_batch + i * stride_U_row + cols * stride_U_col, mask=mask, other=0.0)
            y = tl.load(Y_ptr + pid_batch * stride_Y_batch + pid_rhs * stride_Y_rhs + cols * stride_Y_row, mask=mask, other=0.0)
            sum_val += tl.sum(u * y)
        y_i = tl.load(Y_ptr + pid_batch * stride_Y_batch + pid_rhs * stride_Y_rhs + i * stride_Y_row)
        y_i = (y_i - sum_val) / tl.load(U_ptr + pid_batch * stride_U_batch + i * stride_U_row + i * stride_U_col)
        tl.store(Y_ptr + pid_batch * stride_Y_batch + pid_rhs * stride_Y_rhs + i * stride_Y_row, y_i)

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    # Validate inputs
    assert A.shape[:-2] == Bs.shape[:-2], "Batch dimensions must match"
    n = A.shape[-2]
    assert A.shape[-1] == n, "A must be square"
    k = Bs.shape[-1]
    
    # Compute LU decomposition
    LU, pivots = torch.lu(A, pivot=pivot)
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(Bs)
    else:
        assert out.shape == Bs.shape, "Output tensor shape mismatch"
    
    # Apply permutation using Triton kernel
    if pivot:
        permuted_Bs = torch.empty_like(Bs)
        grid = (A.shape[0], Bs.shape[-1], triton.cdiv(n, 32))
        permute_rhs_kernel[grid](
            Bs, permuted_Bs, pivots,
            n, k,
            Bs.stride(-3), Bs.stride(-1), Bs.stride(-2),
            permuted_Bs.stride(-3), permuted_Bs.stride(-1), permuted_Bs.stride(-2),
            pivots.stride(0),
            BLOCK_SIZE=32
        )
    else:
        permuted_Bs = Bs
    
    # Extract L and U
    L = torch.tril(LU, diagonal=-1) + torch.eye(n, device=A.device)
    U = torch.triu(LU)
    
    # Forward substitution
    y = torch.empty_like(permuted_Bs)
    grid = (A.shape[0], Bs.shape[-1])
    forward_substitution_kernel[grid](
        L, permuted_Bs,
        n, k,
        L.stride(0), L.stride(1), L.stride(2),
        y.stride(0), y.stride(2), y.stride(1),
        BLOCK_SIZE=32
    )
    
    # Backward substitution
    grid = (A.shape[0], Bs.shape[-1])
    backward_substitution_kernel[grid](
        U, y,
        n, k,
        U.stride(0), U.stride(1), U.stride(2),
        out.stride(0), out.stride(2), out.stride(1),
        BLOCK_SIZE=32
    )
    
    return out
