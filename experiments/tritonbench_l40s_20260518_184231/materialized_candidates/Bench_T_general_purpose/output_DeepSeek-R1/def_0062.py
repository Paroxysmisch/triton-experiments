import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot_kernel(y_ptr, x_ptr, alpha, dot_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load y and x
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute scaled addition and update y in-place
    y_updated = y + alpha * x
    tl.store(y_ptr + offsets, y_updated, mask=mask)

    # Compute squares for the dot product and sum them
    squares = y_updated * y_updated
    block_sum = tl.sum(squares, axis=0)

    # Atomically accumulate the sum into the dot product tensor
    tl.atomic_add(dot_ptr, block_sum)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Check inputs
    assert y.is_cuda and x.is_cuda, "Inputs must be CUDA tensors"
    assert y.shape == x.shape, "y and x must have the same shape"
    assert len(y.shape) == 1, "y and x must be 1D tensors"

    n = y.numel()
    # Create a scalar tensor to hold the dot product result
    dot_product = torch.zeros((), device=y.device, dtype=y.dtype)

    # Determine the block size and grid dimensions
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )

    # Launch the kernel
    scaled_add_dot_kernel[grid](y, x, alpha, dot_product, n, BLOCK_SIZE=BLOCK_SIZE)

    # Synchronize to ensure the kernel completes
    torch.cuda.synchronize(device=y.device)

    return dot_product
