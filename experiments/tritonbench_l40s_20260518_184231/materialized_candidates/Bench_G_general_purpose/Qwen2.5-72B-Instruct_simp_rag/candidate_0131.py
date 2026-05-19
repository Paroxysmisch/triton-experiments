import triton
import triton.language as tl

# Triton kernel
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK_SIZE: tl.constexpr):
    # Compute the block index
    pid = tl.program_id(axis=0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to guard memory operations
    mask = offsets < N
    # Load the input tensors
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    # Compute the result
    z = x + y
    # Store the result
    tl.store(Z + offsets, z, mask=mask)

# Triton wrapper
def add_wrapper(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE):
    # Launch the kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    add_kernel[grid](X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
import torch

# Define the block size
BLOCK_SIZE = 128

# Create input tensors
X = torch.randn(1024, device='cuda')
Y = torch.randn(1024, device='cuda')

# Allocate output tensor
Z = torch.empty_like(X)

# Call the wrapper function
add_wrapper(X.data_ptr(), Y.data_ptr(), Z.data_ptr(), X.numel(), BLOCK_SIZE)

# Verify the result
torch.allclose(Z, X + Y)
