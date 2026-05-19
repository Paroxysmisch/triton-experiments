import triton
import triton.language as tl
import torch

# Kernel: y = alpha * (A @ x) + beta * y
@triton.jit
def _matvec_update_kernel(
    A_ptr, x_ptr, y_ptr,
    n, m,
    alpha, beta,
    strideA0, strideA1,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    row_offset = tl.program_id(0) * BLOCK_SIZE_N
    row_indices = row_offset + tl.arange(0, BLOCK_SIZE_N)
    # Each element in row_indices corresponds to a row of A, and an element of y.

    # Initialize partial sums
    partial = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over columns in blocks of BLOCK_SIZE_M
    col_range = tl.arange(0, BLOCK_SIZE_M)
    for cstart in range(0, m, BLOCK_SIZE_M):
        cmask = cstart + col_range
        mask = (cmask < m) & (row_indices < n)
        # Load portion of x
        x_part = tl.load(x_ptr + cmask, mask=(cmask < m), other=0.0)
        # Broadcast x to each row
        x_broadcast = x_part[None, :]

        # Load portion of A
        A_row_offsets = (row_indices[:, None] * strideA0) + (cstart + col_range[None, :]) * strideA1
        A_values = tl.load(A_ptr + A_row_offsets, mask=mask[:, None], other=0.0)

        # Accumulate partial sums: dot row i of A with x
        partial += tl.sum(A_values * x_broadcast, 1)

    # Write back to y with scaling alpha and beta
    mask_y = row_indices < n
    y_val = tl.load(y_ptr + row_indices, mask=mask_y, other=0.0)
    y_val = alpha * partial + beta * y_val
    tl.store(y_ptr + row_indices, y_val, mask=mask_y)


# Kernel: partial reduction for dot(y, x)
@triton.jit
def _dot_partial_kernel(
    y_ptr, x_ptr, partial_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    y_val = tl.load(y_ptr + offset, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + (offset % n), mask=mask, other=0.0)
    acc = y_val * x_val
    # Reduce within the block
    for stride in [16, 8, 4, 2, 1]:
        acc += tl.swizzle2d(acc, [stride, 0])
    # Store one element per block
    if tl.thread_idx.x == 0:
        tl.store(partial_ptr + pid, acc[0])


def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    y = alpha * torch.mv(A, x) + beta * y
    result = torch.dot(y, x)
    """
    # Shapes
    n, m = A.shape
    # Ensure CUDA tensors
    A_ptr = A.contiguous().cuda()
    x_ptr = x.contiguous().cuda()
    y_ptr = y.contiguous().cuda()

    # Launch matrix-vector update
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128
    grid = ( (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, )
    _matvec_update_kernel[grid](
        A_ptr, x_ptr, y_ptr,
        n, m,
        alpha, beta,
        A_ptr.stride(0), A_ptr.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M
    )

    # Dot product partial
    BLOCK_SIZE = 256
    grid_dot = ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    partial_ptr = torch.empty(grid_dot[0], dtype=A.dtype, device=A.device)
    _dot_partial_kernel[grid_dot](
        y_ptr, x_ptr, partial_ptr,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Sum partial results
    result = partial_ptr.sum()

    # Copy updated y back
    y.copy_(y_ptr)

    return result
