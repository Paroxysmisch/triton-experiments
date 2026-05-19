import torch
import triton
import triton.language as tl

# Triton kernel for ReLU activation
@triton.jit
def relu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(axis=0)
    # Block start index
    block_start = pid * BLOCK_SIZE
    # Offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load the input block from global memory
    x = tl.load(X + offsets, mask=offsets < N, other=0.0)
    # Apply the ReLU activation
    y = tl.where(x >= 0, x, 0.0)
    # Store the result back to global memory
    tl.store(Y + offsets, y, mask=offsets < N)

# Wrapper function to launch the kernel
def relu(X: torch.Tensor):
    # Ensure the input tensor is on the GPU
    assert X.is_cuda
    # Output tensor with the same shape and type as the input
    Y = torch.empty_like(X)
    # Number of elements in the tensor
    N = X.numel()
    # Block size for the kernel
    BLOCK_SIZE = 1024
    # Number of blocks needed to cover the entire tensor
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel on a grid of blocks
    relu_kernel[(num_blocks,)](X, Y, N, BLOCK_SIZE)
    # Return the ReLU-activated tensor
    return Y
