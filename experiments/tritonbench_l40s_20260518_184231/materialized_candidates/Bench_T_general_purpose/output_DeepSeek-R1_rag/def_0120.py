import torch
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(
    A_ptr, x_ptr, y_ptr, alpha, beta,
    n, m,
    stride_an, stride_am,
    BLOCK_SIZE_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_ROWS
    row_end = min(row_start + BLOCK_SIZE_ROWS, n)

    for row_idx in range(row_start, row_end):
        sum_acc = tl.zeros((1,), dtype=tl.float32)
        for col_block in range(0, m, BLOCK_SIZE_COLS):
            cols = col_block + tl.arange(0, BLOCK_SIZE_COLS)
            mask = cols < m
            a_ptrs = A_ptr + row_idx * stride_an + cols * stride_am
            a = tl.load(a_ptrs, mask=mask, other=0.0)
            x = tl.load(x_ptr + cols, mask=mask, other=0.0)
            sum_acc += tl.sum(a * x)
        current_y = tl.load(y_ptr + row_idx)
        new_y = alpha * sum_acc + beta * current_y
        tl.store(y_ptr + row_idx, new_y)

@triton.jit
def dot_kernel(
    y_ptr, x_ptr, output_ptr,
    n,
    stride_y,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    y = tl.load(y_ptr + offs * stride_y, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs * stride_x, mask=mask, other=0.0)
    acc = tl.sum(y * x)
    tl.atomic_add(output_ptr, acc)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    assert x.shape == (m,), f"x must be of shape ({m},), but got {x.shape}"
    assert y.shape == (n,), f"y must be of shape ({n},), but got {y.shape}"
    assert m == n, f"m must equal n for dot product, but got m={m}, n={n}"

    # Launch GEMV kernel to update y in-place
    BLOCK_SIZE_ROWS = 4
    BLOCK_SIZE_COLS = 128
    grid = (triton.cdiv(n, BLOCK_SIZE_ROWS),)
    gemv_kernel[grid](
        A, x, y, alpha, beta,
        n, m,
        A.stride(0), A.stride(1),
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
    )

    # Compute the dot product of updated y and x
    output = torch.zeros(1, device=y.device, dtype=torch.float32)
    BLOCK_SIZE_DOT = 1024
    grid_dot = (triton.cdiv(n, BLOCK_SIZE_DOT),)
    dot_kernel[grid_dot](
        y, x, output,
        n,
        y.stride(0), x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE_DOT,
    )

    return output.squeeze()
