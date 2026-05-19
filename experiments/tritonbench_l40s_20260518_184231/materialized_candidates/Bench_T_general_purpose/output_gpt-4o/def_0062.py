import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot_kernel(
    y_ptr, x_ptr, alpha, result_ptr, n, BLOCK_SIZE: tl.constexpr
):
    # Calculate the block and thread index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)
    y = tl.load(y_ptr + offsets, mask=offsets < n, other=0.0)

    # Compute y += alpha * x
    y = y + alpha * x

    # Store the updated y back to global memory
    tl.store(y_ptr + offsets, y, mask=offsets < n)

    # Compute the partial dot product
    partial_dot = tl.sum(y * y, axis=0)

    # Atomically add the partial result to the result buffer
    tl.atomic_add(result_ptr, partial_dot)


def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure y and x are contiguous
    y = y.contiguous()
    x = x.contiguous()

    # Check that y and x have the same shape
    assert y.shape == x.shape, "y and x must have the same shape"

    # Allocate a buffer for the result
    result = torch.zeros(1, dtype=y.dtype, device=y.device)

    # Define the block size and grid size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)

    # Launch the Triton kernel
    scaled_add_dot_kernel[grid](
        y, x, alpha, result, y.numel(), BLOCK_SIZE=BLOCK_SIZE
    )

    return result


# Example usage:
y = torch.tensor([1.0, 2.0, 3.0], device='cuda')
x = torch.tensor([4.0, 5.0, 6.0], device='cuda')
alpha = 0.5

result = scaled_add_dot(y, x, alpha)
print("Modified y:", y)
print("Dot product of modified y with itself:", result.item())
