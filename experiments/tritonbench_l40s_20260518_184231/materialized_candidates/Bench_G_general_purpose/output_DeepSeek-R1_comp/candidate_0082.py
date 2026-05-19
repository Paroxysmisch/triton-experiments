import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_rows = rows < M

    max_values = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')

    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask_cols = cols < N
        mask = mask_rows[:, None] & mask_cols[None, :]

        x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
        current_block = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        current_max = tl.max(current_block, axis=1)
        max_values = tl.maximum(max_values, current_max)

    y_ptrs = y_ptr + rows * stride_y
    tl.store(y_ptrs, max_values, mask=mask_rows)

def load_reduce(x: torch.Tensor):
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    load_reduce_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y

# Test case
torch.manual_seed(0)
x = torch.randn(512, 256, device='cuda')
y_triton = load_reduce(x)
y_torch = x.max(dim=1)[0]

assert torch.allclose(y_triton, y_torch, atol=1e-6), "Triton and Torch outputs differ"
print("Test passed! Outputs match.")
