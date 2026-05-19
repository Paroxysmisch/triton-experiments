import triton
import triton.language as tl

# Kernel 1: Processes a block of the input tensor, finding the local maximum and its index in the block.
@triton.jit
def argmax_kernel_1(X, X_size, max_val, max_idx, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X_size

    x = tl.load(X + offsets, mask=mask, other=-float('inf'))
    max_val_local = tl.max(x, axis=0)
    max_idx_local = tl.argmax(x, axis=0)

    tl.store(max_val + pid, max_val_local)
    tl.store(max_idx + pid, max_idx_local)

# Kernel 2: Aggregates the results from argmax_kernel_1 to find the overall maximum value's index.
@triton.jit
def argmax_kernel_2(max_val, max_idx, max_val_out, max_idx_out, num_blocks: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid == 0:
        max_val_local = tl.load(max_val, mask=tl.arange(0, num_blocks) < num_blocks, other=-float('inf'))
        max_idx_local = tl.argmax(max_val_local, axis=0)
        max_val_global = tl.max(max_val_local, axis=0)

        tl.store(max_val_out, max_val_global)
        tl.store(max_idx_out, max_idx_local)

# Kernel 3: Handles multi-dimensional tensors, finding the maximum index along a specified dimension.
@triton.jit
def argmax_kernel(X, X_size, max_val_out, max_idx_out, stride, dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X_size

    x = tl.load(X + offsets * stride, mask=mask, other=-float('inf'))
    max_val_local = tl.max(x, axis=0)
    max_idx_local = tl.argmax(x, axis=0)

    tl.store(max_val_out + pid, max_val_local)
    tl.store(max_idx_out + pid, max_idx_local)

# Wrapper function to handle the argmax operation
def argmax(x, dim=None):
    if dim is None:
        # Flatten the tensor and find the argmax
        x_size = x.size
        max_val = triton.empty((1,), device=x.device, dtype=x.dtype)
        max_idx = triton.empty((1,), device=x.device, dtype=tl.int32)

        # Step 1: Process blocks
        num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmax_kernel_1[(num_blocks,)](x, x_size, max_val, max_idx, 1, BLOCK_SIZE=BLOCK_SIZE)

        # Step 2: Aggregate results
        argmax_kernel_2[(1,)](max_val, max_idx, max_val, max_idx, num_blocks)

        return max_idx.item()
    else:
        # Handle multi-dimensional tensor
        shape = x.shape
        dim_size = shape[dim]
        other_size = x.size // dim_size
        max_val_out = triton.empty((other_size,), device=x.device, dtype=x.dtype)
        max_idx_out = triton.empty((other_size,), device=x.device, dtype=tl.int32)

        # Step 1: Process blocks along the specified dimension
        num_blocks = (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        argmax_kernel[(other_size,)](x, dim_size, max_val_out, max_idx_out, other_size, dim, BLOCK_SIZE=BLOCK_SIZE)

        return max_idx_out

# Example usage
if __name__ == "__main__":
    import torch

    # Example tensor
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0], device='cuda')

    # Find argmax of the entire tensor
    max_idx = argmax(x)
    print(f"Max index of the entire tensor: {max_idx}")

    # Find argmax along a specific dimension
    x_2d = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]], device='cuda')
    max_idx_2d = argmax(x_2d, dim=1)
    print(f"Max indices along dimension 1: {max_idx_2d}")
