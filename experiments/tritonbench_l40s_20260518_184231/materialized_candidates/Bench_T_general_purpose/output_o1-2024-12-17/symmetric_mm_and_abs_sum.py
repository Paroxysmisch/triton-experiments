import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_kernel(
    A_ptr, C_ptr,
    n, m,
    alpha, beta,
    stride_a, stride_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Each program handles a [BLOCK_M, BLOCK_N] block of the output.
    For symmetry: (A * A^T)[i, j] = sum_k(A[i, k] * A[j, k]), i, j in [0, n)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block indices
    row_off = pid_m * BLOCK_M
    col_off = pid_n * BLOCK_N

    # Create accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k dimension in sub-blocks of size BLOCK_K
    for k in range(0, m, BLOCK_K):
        # Load A block
        a_row = row_off + tl.arange(0, BLOCK_M)[:, None]
        a_col = k + tl.arange(0, BLOCK_K)[None, :]
        # Load transposed portion from A as well (mapping col_off -> row in A)
        a_t_row = col_off + tl.arange(0, BLOCK_N)[None, :]
        a_t_col = k + tl.arange(0, BLOCK_K)[:, None]

        a_mask = (a_row < n) & (a_col < m)
        a_t_mask = (a_t_row < n) & (a_t_col < m)

        # A[i, k] shape: [BLOCK_M, BLOCK_K]
        # A[j, k] shape: [BLOCK_N, BLOCK_K] but we must read it as row=j, col=k
        # So we gather from row=a_t_row, col=a_t_col
        a_block = tl.load(A_ptr + a_row * stride_a + a_col, mask=a_mask, other=0.0)
        a_t_block = tl.load(A_ptr + a_t_row * stride_a + a_t_col, mask=a_t_mask, other=0.0)

        # transposing the second block for elementwise multiply:
        # we have a_t_block with shape [BLOCK_N, BLOCK_K], so transpose it to [BLOCK_K, BLOCK_N]
        a_t_block_t = tl.transpose(a_t_block, 0, 1)

        # compute product and accumulate
        acc += tl.dot(a_block, a_t_block_t)

    # Scale by alpha and add beta*C
    c_row = row_off + tl.arange(0, BLOCK_M)[:, None]
    c_col = col_off + tl.arange(0, BLOCK_N)[None, :]
    c_mask = (c_row < n) & (c_col < n)
    c_old = tl.load(C_ptr + c_row * stride_c + c_col, mask=c_mask, other=0.0)
    c_new = alpha * acc + beta * c_old

    # Write back
    tl.store(C_ptr + c_row * stride_c + c_col, c_new, mask=c_mask)

@triton.jit
def _abs_sum_kernel(
    C_ptr,
    PARTIALS_ptr,
    n,
    stride_c,
    BLOCK_SIZE: tl.constexpr
):
    """
    Each program reduces one block of C along the 2D dimension into a single partial sum.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    row_end = tl.min(row_start + BLOCK_SIZE, n)

    # Accumulator for partial sum
    partial_sum = 0.0
    for row in range(row_start, row_end):
        c_row_ptr = C_ptr + row * stride_c
        # We iterate over columns in steps for better parallelization
        col = 0
        while col < n:
            mask = col + tl.arange(0, 32) < n
            vals = tl.load(c_row_ptr + (col + tl.arange(0, 32)), mask=mask, other=0.0)
            partial_sum += tl.sum(tl.abs(vals), 0)
            col += 32

    tl.store(PARTIALS_ptr + pid, partial_sum)

@triton.jit
def _final_sum_kernel(PARTIALS_ptr, RES_ptr, num_warps: tl.constexpr):
    """
    Single-warp kernel to reduce partial sums into a single scalar.
    """
    partial_sum = 0.0
    for i in range(num_warps):
        partial_sum += tl.load(PARTIALS_ptr + i)
    # Store final result
    tl.store(RES_ptr, partial_sum)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    A: (n, m)
    C: (n, n)
    Output:
       1) C = alpha * (A @ A.T) + beta * C
       2) asum = sum(abs(C))
    Returns a scalar tensor with asum.
    """
    assert A.is_cuda and C.is_cuda, "Tensors must be on CUDA."
    n, m = A.shape
    assert C.shape[0] == n and C.shape[1] == n, "C must be (n, n)."

    # Launch kernel for symmetric mm
    BLOCK = 128
    grid = ( (n + BLOCK - 1) // BLOCK, (n + BLOCK - 1) // BLOCK )
    _symmetric_mm_kernel[grid](
        A, C,
        n, m,
        alpha, beta,
        A.stride(0), C.stride(0),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=32
    )

    # Prepare partial buffer for abs-sum
    # Each block reduces a slice of rows
    BLOCK_SIZE = 128
    num_warps = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    partials = torch.empty(num_warps, dtype=torch.float32, device=A.device)

    _abs_sum_kernel[num_warps](
        C,
        partials,
        n,
        C.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Final reduce
    result_tensor = torch.empty(1, dtype=torch.float32, device=A.device)
    _final_sum_kernel[1](
        partials,
        result_tensor,
        num_warps
    )

    return result_tensor.clone()  # return scalar tensor
