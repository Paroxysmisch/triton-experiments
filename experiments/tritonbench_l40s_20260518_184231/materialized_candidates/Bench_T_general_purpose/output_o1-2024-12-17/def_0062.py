import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_add_dot_kernel(
    y_ptr, x_ptr, out_ptr,
    alpha, n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    y_val = tl.where(mask, tl.load(y_ptr + offsets), 0.0)
    x_val = tl.where(mask, tl.load(x_ptr + offsets), 0.0)

    scaled_x = x_val * alpha
    new_y = y_val + scaled_x

    tl.store(y_ptr + offsets, new_y, mask=mask)

    part = new_y * new_y
    part_sum = tl.sum(part, axis=0)
    tl.atomic_add(out_ptr, part_sum)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    n = y.numel()
    out = torch.zeros(1, device=y.device, dtype=y.dtype)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _scaled_add_dot_kernel[grid](y, x, out, alpha, n, BLOCK_SIZE=BLOCK_SIZE)

    return out
