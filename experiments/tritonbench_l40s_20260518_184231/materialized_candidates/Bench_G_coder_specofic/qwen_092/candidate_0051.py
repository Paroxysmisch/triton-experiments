import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 256

@triton.jit
def max_kernel_1(x, mid, pid, BLOCK_SIZE):
    """
    Computes the maximum values within a large 1D input tensor across predefined blocks.
    """
    # Get the index within the block
    tid = tl.program_id(0)
    # Calculate the starting index for the block
    start_idx = pid * BLOCK_SIZE + tid
    # Load values with masking to handle out-of-bound reads
    x_val = tl.load(x + start_idx, mask=start_idx < x.shape[0], other=-float('inf'))
    # Compute the maximum value in the block
    mid_val = tl.max(x_val)
    # Store the maximum value in the mid tensor
    tl.store(mid + pid, mid_val)

@triton.jit
def max_kernel_2(mid, out, num_blocks):
    """
    Consolidates results from max_kernel_1 to calculate the overall maximum.
    """
    # Get the index within the block
    tid = tl.program_id(0)
    # Calculate the starting index for the block
    start_idx = tid * BLOCK_SIZE
    # Load values with masking to handle out-of-bound reads
    mid_val = tl.load(mid + start_idx, mask=start_idx < num_blocks, other=-float('inf'))
    # Compute the maximum value in the block
    out_val = tl.max(mid_val)
    # Store the maximum value in the out tensor
    tl.store(out, out_val)

@triton.jit
def max_kernel(x, out, pid_m, pid_k, BLOCK_SIZE, dim):
    """
    Computes the maximum values along a specified dimension.
    """
    # Get the index within the block
    tid = tl.program_id(0)
    # Calculate the starting index for the block
    start_idx_m = pid_m * BLOCK_SIZE + tid
    start_idx_k = pid_k * BLOCK_SIZE + tid
    # Load values with masking to handle out-of-bound reads
    x_val = tl.load(x + start_idx_m * x.stride(0) + start_idx_k * x.stride(1), mask=start_idx_m < x.shape[0] and start_idx_k < x.shape[1], other=-float('inf'))
    # Compute the maximum value and index across dimensions
    max_val, max_idx = tl.max_with_index(x_val)
    # Store the maximum value in the out tensor
    tl.store(out + start_idx_m * out.stride(0) + max_idx * out.stride(1), max_val)

def max(x):
    """
    Wrapper function to execute the sequential execution of max_kernel_1 and max_kernel_2.
    """
    # Prepare intermediate tensors
    mid = tl.zeros(x.shape[0] // BLOCK_SIZE, dtype=tl.float32)
    out = tl.zeros(1, dtype=tl.float32)
    # Determine optimal block size
    grid_size = (x.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Execute max_kernel_1
    max_kernel_1[x.shape[0] // BLOCK_SIZE](x, mid, tl.arange(x.shape[0] // BLOCK_SIZE), BLOCK_SIZE)
    # Execute max_kernel_2
    max_kernel_2[mid.shape[0] // BLOCK_SIZE](mid, out, mid.shape[0] // BLOCK_SIZE)
    return out[0]

def max_dim(x, dim):
    """
    Extends max computation to operate on any dimension specified by dim.
    """
    if dim < 0 or dim >= x.ndim:
        raise ValueError("Invalid dimension index")
    # Reshape outputs accordingly
    out_shape = x.shape[:dim] + (1,) + x.shape[dim+1:]
    out = tl.zeros(out_shape, dtype=tl.float32)
    # Calculate grid dimensions
    M = x.shape[0] // BLOCK_SIZE
    K = x.shape[1] // BLOCK_SIZE
    # Execute max_kernel
    max_kernel[x.shape[0] // BLOCK_SIZE, x.shape[1] // BLOCK_SIZE](x, out, tl.arange(M), tl.arange(K), BLOCK_SIZE, dim)
    return out

# Example usage
x = tl.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=tl.float32)
max_val = max(x)
print("Max value:", max_val)

x_dim = tl.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=tl.float32)
max_val_dim = max_dim(x_dim, 1)
print("Max values along dim 1:", max_val_dim)
