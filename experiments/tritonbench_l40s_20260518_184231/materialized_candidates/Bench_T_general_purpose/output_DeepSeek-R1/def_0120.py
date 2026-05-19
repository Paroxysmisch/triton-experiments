import torch
import triton
import triton.language as tl

@triton.jit
def update_y_kernel(
    A_ptr, x_ptr, y_ptr, alpha, beta,
    n, m,
    stride_am, stride_an,
    stride_x,
    stride_y,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n:
        return
    sum_acc = tl.zeros([1], dtype=tl.float32)
    for j in range(0, m, BLOCK_SIZE):
        col_offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < m
        a_ptrs = A_ptr + row_idx * stride_an + col_offsets * stride_am
        x_ptrs = x_ptr + col_offsets * stride_x
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        x_val = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_acc += tl.sum(a * x_val)
    sum_acc *= alpha
    y_ptr_row = y_ptr + row_idx * stride_y
    y_val = tl.load(y_ptr_row)
    new_y_val = sum_acc + beta * y_val
    tl.store(y_ptr_row, new_y_val)

@triton.jit
def dot_kernel(
    y_ptr, x_ptr, output_ptr,
    num_elements,
    stride_y,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    y = tl.load(y_ptr + offsets * stride_y, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets * stride_x, mask=mask, other=0.0)
    product = y * x
    sum_result = tl.sum(product, axis=0)
    tl.atomic_add(output_ptr, sum_result)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    assert x.shape == (m,), f"Expected x to have shape ({m},), got {x.shape}"
    assert y.shape == (n,), f"Expected y to have shape ({n},), got {y.shape}"
    assert A.is_cuda and x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert n == m, f"Dot product requires n == m, got n={n}, m={m}"

    BLOCK_SIZE = 128
    grid = (n,)
    update_y_kernel[grid](
        A.data_ptr(), x.data_ptr(), y.data_ptr(), alpha, beta,
        n, m,
        A.stride(1), A.stride(0),
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output = torch.zeros(1, device=y.device, dtype=y.dtype)
    num_elements = n
    grid_dot = (triton.cdiv(num_elements, BLOCK_SIZE),)
    dot_kernel[grid_dot](
        y.data_ptr(), x.data_ptr(), output.data_ptr(),
        num_elements,
        y.stride(0), x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output[0]
