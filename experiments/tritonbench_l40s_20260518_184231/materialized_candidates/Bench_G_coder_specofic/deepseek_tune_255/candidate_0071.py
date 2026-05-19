import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(
    x_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    x_row_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(BLOCK_SIZE, n_cols),
        strides=(x_row_stride, 1),
        offsets=(row_idx * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, n_cols),
        order=(1, 0),
    )
    y_row_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(BLOCK_SIZE, n_cols),
        strides=(y_row_stride, 1),
        offsets=(row_idx * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, n_cols),
        order=(1, 0),
    )
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x_mask = col_offsets < n_cols
    x = tl.load(x_row_ptr, mask=x_mask, other=0.0)
    y = x * x
    tl.store(y_row_ptr, y, mask=x_mask)

def square(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    square_kernel[(n_rows,)](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

torch.manual_seed(42)
x = torch.randn(1800, 1800)
triton_out = square(x)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Input")
plt.imshow(x.abs())
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title("Triton Output")
plt.imshow(triton_out.abs())
plt.colorbar()
plt.show()
