import triton
import triton.language as tl
import torch

@triton.jit
def _scaled_add_dot_kernel(
    x_ptr: tl.pointer_type elementtype=tl.float32,
    y_ptr: tl.pointer_type elementtype=tl.float32,
    partial_sums_ptr: tl.pointer_type elementtype=tl.float32,
    alpha: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.where(mask, tl.load(x_ptr + offsets), 0.0)
    y = tl.where(mask, tl.load(y_ptr + offsets), 0.0)
    y = y + alpha * x
    tl.store(y_ptr + offsets, y, mask=mask)

    sq = y * y
    partial_sum = tl.sum(sq, axis=0)
    tl.store(partial_sums_ptr + pid, partial_sum)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    n = y.numel()
    assert x.numel() == n, "x and y must have the same number of elements"
    assert y.is_cuda and x.is_cuda, "Tensors must be on CUDA device"
    BLOCK_SIZE = 1024
    grid = ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    partial_sums = torch.zeros(grid[0], dtype=y.dtype, device=y.device)
    _scaled_add_dot_kernel[grid](x, y, partial_sums, alpha, n, BLOCK_SIZE=BLOCK_SIZE)
    return partial_sums.sum()
