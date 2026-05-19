import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mv_kernel(
    A_ptr, x_ptr, alpha, beta, y_ptr,
    n, m,
    stride_am, stride_ak,
    x_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n:
        return

    sum_acc = tl.zeros((1,), dtype=tl.float32)
    for k in range(0, m, BLOCK_SIZE):
        col_offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < m
        a_ptrs = A_ptr + row_idx * stride_am + col_offsets * stride_ak
        x_ptrs = x_ptr + col_offsets * x_stride

        a = tl.load(a_ptrs, mask=mask, other=0.0)
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_acc += tl.sum(a * x)

    x_val = tl.load(x_ptr + row_idx * x_stride)
    y_val = alpha * sum_acc + beta * x_val
    tl.store(y_ptr + row_idx, y_val)

@triton.jit
def norm_reduction_kernel(
    y_ptr, norm_ptr, p, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    abs_y = tl.abs(y)
    powered = tl.pow(abs_y, p)
    block_sum = tl.sum(powered, axis=0)

    tl.atomic_add(norm_ptr, block_sum)

def symmetric_matrix_vector_norm(
    A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0
) -> torch.Tensor:
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    n = A.size(0)
    assert x.dim() == 1 and x.size(0) == n, "x must be a vector of size n"

    if not A.is_contiguous():
        A = A.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    y = torch.empty_like(x)

    grid = lambda meta: (n,)
    BLOCK_SIZE = 128
    symmetric_mv_kernel[grid](
        A, x, alpha, beta, y,
        n, n,
        A.stride(0), A.stride(1),
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    sum_powered = torch.zeros(1, dtype=y.dtype, device=y.device)
    num_elements = y.numel()
    grid_norm = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    norm_reduction_kernel[grid_norm](
        y, sum_powered, p, num_elements,
        BLOCK_SIZE=1024
    )

    sum_val = sum_powered.item()
    if p == 0.0:
        raise ValueError("p=0 not supported for norm calculation")
    if sum_val == 0 and p < 0:
        raise ValueError("Negative p requires non-zero elements")
    norm_val = sum_val ** (1.0 / p) if p != 0 else 0.0
    return torch.tensor(norm_val, dtype=y.dtype, device=y.device)
