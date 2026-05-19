import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    D1, D2, N, stride,
    grid_stride, block_stride,
    grid_stride_W, block_stride_W,
    BLOCK_SIZE
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE
    x_offsets = offsets * stride
    w_offsets = offsets * block_stride_W

    # Load weights and input into shared memory
    X = tl.load(X_ptr + x_offsets, BLOCK_SIZE)
    W = tl.load(W_ptr + w_offsets, BLOCK_SIZE)

    # Compute mean and variance
    mean = tl.sum(X, axis=2) / N
    var = tl.sum((X - mean[:, :, None]) ** 2, axis=2) / N
    std = tl.sqrt(var)

    # Normalize input
    X_norm = (X - mean[:, :, None]) / std[:, :, None]

    # Apply weights and write output
    Y = X_norm * W[None, :, :]
    tl.store(Y_ptr + offsets, Y)

def layernorm_forward(X, W, Y, stride):
    # Compute grid size
    grid_size = (D2 - 1) // BLOCK_SIZE + 1

    # Compute memory strides
    grid_stride = grid_size * BLOCK_SIZE
    block_stride = BLOCK_SIZE
    grid_stride_W = grid_size * block_stride_W
    block_stride_W = block_stride_W

    # Check dimensions
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == Y.shape[1]
    assert X.shape[2] == W.shape[1]

    # Invoke kernel
    _layer_norm_fwd_kernel[grid_size, BLOCK_SIZE](
        X.device_ctypes_ptr,
        W.device_ctypes_ptr,
        Y.device_ctypes_ptr,
        X.shape[0], X.shape[1], X.shape[2],
        stride,
        grid_stride, block_stride,
        grid_stride_W, block_stride_W,
        BLOCK_SIZE
    )
