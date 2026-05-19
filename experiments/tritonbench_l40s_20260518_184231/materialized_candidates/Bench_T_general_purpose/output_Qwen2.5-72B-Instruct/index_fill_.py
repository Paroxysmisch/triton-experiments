import triton
import triton.language as tl

@triton.jit
def index_fill_kernel(
    X,  # Pointer to the input tensor
    X_size_0,  # Size of the first dimension of X
    X_size_1,  # Size of the second dimension of X
    index,  # Pointer to the index tensor
    index_size,  # Size of the index tensor
    value,  # Value to fill
    dim,  # Dimension along which to index
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X_size_0 if dim == 0 else offsets < X_size_1

    x_offsets = offsets if dim == 0 else tl.arange(0, X_size_1)
    y_offsets = tl.arange(0, X_size_1) if dim == 0 else offsets

    x_offsets = x_offsets[:, None] if dim == 0 else x_offsets[None, :]
    y_offsets = y_offsets[:, None] if dim == 1 else y_offsets[None, :]

    for i in range(index_size):
        idx = tl.load(index + i)
        x_idx = x_offsets if dim == 0 else idx
        y_idx = idx if dim == 0 else y_offsets

        X_ptr = X + x_idx * X_size_1 + y_idx
        tl.store(X_ptr, value, mask=mask)

import torch
import triton
import triton.language as tl

def index_fill_(X, dim, index, value):
    # Convert index to a 1D tensor if it's not already
    index = index.flatten()

    # Get the sizes of the tensor and the index tensor
    X_size_0, X_size_1 = X.shape
    index_size = index.numel()

    # Allocate the grid and block sizes
    grid = (triton.cdiv(X_size_0 if dim == 0 else X_size_1, 1024),)
    block = 1024

    # Launch the kernel
    index_fill_kernel[grid](
        X.data_ptr(), X_size_0, X_size_1, index.data_ptr(), index_size, value, dim, block
    )

    return X

# Example usage
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 2])
x = index_fill_(x, 1, index, -1)
print(x)
