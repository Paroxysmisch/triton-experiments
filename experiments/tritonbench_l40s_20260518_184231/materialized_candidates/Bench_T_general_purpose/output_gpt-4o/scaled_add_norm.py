import triton
import triton.language as tl

@triton.jit
def scaled_add_kernel(y_ptr, x_ptr, alpha, n, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current element
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load elements from y and x
    y = tl.load(y_ptr + offsets, mask=offsets < n, other=0.0)
    x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)

    # Perform the operation y += alpha * x
    y += alpha * x

    # Store the result back to y
    tl.store(y_ptr + offsets, y, mask=offsets < n)

import torch

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure y and x are on the same device and have the same shape
    assert y.shape == x.shape, "Tensors y and x must have the same shape"
    assert y.device == x.device, "Tensors y and x must be on the same device"

    # Number of elements in the tensor
    n = y.numel()

    # Define the block size for the Triton kernel
    BLOCK_SIZE = 1024

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    scaled_add_kernel[grid](y, x, alpha, n, BLOCK_SIZE=BLOCK_SIZE)

    # Calculate the 2-norm of the modified y using PyTorch
    norm = torch.norm(y, p=2)

    return norm
