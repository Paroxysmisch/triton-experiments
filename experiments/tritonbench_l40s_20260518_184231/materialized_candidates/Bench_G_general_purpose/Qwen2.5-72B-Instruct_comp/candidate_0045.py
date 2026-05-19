import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    X,  # Pointer to the input tensor
    Y,  # Pointer to the output tensor
    scalar,  # Scalar value to raise each element to
    N,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size (number of elements processed by each block)
):
    # Compute the block index and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input elements into a Triton block
    x = tl.load(X + offsets, mask=offsets < N, other=0.0)

    # Compute the power of each element
    y = tl.pow(x, scalar)

    # Store the results back to the output tensor
    tl.store(Y + offsets, y, mask=offsets < N)

import torch
import triton
import triton.runtime

def pow_func_scalar_tensor_wrapper_rank_1(x, scalar, y=None):
    # Ensure input is a torch.Tensor
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input x must be a torch.Tensor")

    # Ensure output is a torch.Tensor or None
    if y is not None and not isinstance(y, torch.Tensor):
        raise TypeError("Output y must be a torch.Tensor or None")

    # Ensure input and output tensors are on the same device
    if y is not None and x.device != y.device:
        raise ValueError("Input and output tensors must be on the same device")

    # Ensure input and output tensors have the same shape
    if y is not None and x.shape != y.shape:
        raise ValueError("Input and output tensors must have the same shape")

    # Create output tensor if not provided
    if y is None:
        y = torch.empty_like(x)

    # Get the number of elements in the tensor
    N = x.numel()

    # Determine the optimal block size and grid size
    BLOCK_SIZE = 1024
    GRID_SIZE = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[GRID_SIZE, BLOCK_SIZE](
        x, y, scalar, N, BLOCK_SIZE
    )

    return y
