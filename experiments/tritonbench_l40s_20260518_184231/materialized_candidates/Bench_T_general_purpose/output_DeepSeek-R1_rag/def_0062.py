import triton
import triton.language as tl
import torch

@triton.jit
def scaled_add_dot_kernel(
    y_ptr,
    x_ptr,
    alpha,
    dot_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute scaled addition and store back to y
    new_y = y + alpha * x
    tl.store(y_ptr + offsets, new_y, mask=mask)

    # Compute squares and accumulate
    squares = new_y * new_y
    block_sum = tl.sum(squares, axis=0)
    tl.atomic_add(dot_ptr, block_sum)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.dim() == 1 and x.dim() == 1, "Inputs must be 1D tensors"
    assert y.size(0) == x.size(0), "Inputs must have the same length"
    n_elements = y.numel()
    # Allocate output tensor for the dot product (scalar)
    dot = torch.zeros(1, device=y.device, dtype=y.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scaled_add_dot_kernel[grid](y, x, alpha, dot, n_elements, BLOCK_SIZE=1024)
    return dot
