import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 256

@triton.jit
def _layer_norm_fwd_kernel(
    X, W, Y, X_mean, X_var, stride_d1, stride_d2, stride_n, stride_w, D1, D2, N
):
    # Get the indices for the current block
    d1 = tl.program_id(0)
    d2 = tl.program_id(1)
    n = tl.program_id(2)

    # Compute the starting index for the current block
    x_idx = d1 * stride_d1 + d2 * stride_d2 + n * stride_n

    # Initialize the mean and variance accumulators
    mean = 0.0
    var = 0.0

    # Loop over the normalization axis
    for i in range(BLOCK_SIZE):
        # Compute the index within the block
        idx = x_idx + i * stride_n

        # Load the input value
        x_val = tl.load(X + idx, mask=i < N, other=0.0)

        # Compute the mean and variance
        mean += x_val
        var += x_val * x_val

    # Compute the mean and variance for the entire block
    mean /= N
    var /= N - 1.0

    # Synchronize threads within the block to ensure the mean and variance are ready
    tl.barrier()

    # Store the mean and variance in shared memory
    if tl.program_id(2) == 0:
        tl.store(X_mean + d1 * stride_d1 + d2 * stride_d2, mean)
        tl.store(X_var + d1 * stride_d1 + d2 * stride_d2, var)

    # Synchronize threads to ensure the mean and variance are stored
    tl.barrier()

    # Compute the normalized output
    mean = tl.load(X_mean + d1 * stride_d1 + d2 * stride_d2)
    var = tl.load(X_var + d1 * stride_d1 + d2 * stride_d2)
    eps = 1e-5
    x_val = tl.load(X + x_idx, mask=i < N, other=0.0)
    y_val = (x_val - mean) / tl.sqrt(var + eps) * tl.load(W + d2 * stride_w + n, mask=i < N, other=0.0)

    # Store the normalized output
    tl.store(Y + x_idx, y_val, mask=i < N)

@triton.jit
def layernorm_forward(X, W, Y, stride_d1, stride_d2, stride_n, stride_w, D1, D2, N):
    # Compute the grid dimensions
    grid_d1 = (D1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_d2 = (D2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Compute the shared memory strides
    shared_mean_stride = BLOCK_SIZE
    shared_var_stride = BLOCK_SIZE

    # Allocate shared memory for mean and variance
    X_mean = tl.zeros((D1, D2), dtype=tl.float32)
    X_var = tl.zeros((D1, D2), dtype=tl.float32)

    # Launch the kernel
    _layer_norm_fwd_kernel[X_mean, X_var, shared_mean_stride, shared_var_stride](
        X, W, Y, X_mean, X_var, stride_d1, stride_d2, stride_n, stride_w, D1, D2, N
    )

# Example usage
# Assuming X, W, Y are already allocated and initialized on the GPU
# stride_d1, stride_d2, stride_n, stride_w, D1, D2, N are computed based on the tensor shapes
# layernorm_forward(X, W, Y, stride_d1, stride_d2, stride_n, stride_w, D1, D2, N)
