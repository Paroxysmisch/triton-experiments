@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr, Y_ptr, stride_x_row, N, eps, BLOCK_N, M, out_options
):
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)
    if row < M and col < N:
        # Load a block of data from X
        x = tl.load(X_ptr + row * stride_x_row + col)
        # Calculate the sum of squares for variance
        var = tl.dot(x, x)
        # Compute the reciprocal of the square root of the variance plus eps
        rstd = 1.0 / tl.sqrt(var + eps)
        # Multiply the input block by rstd to produce the normalized values
        y = x * rstd
        # Store the normalized values in Y
        tl.store(Y_ptr + row * stride_x_row + col, y)

def _l2_norm_fwd(x, eps=1e-12):
    # Reshape and possibly make the input tensor x contiguous
    x = x.reshape(-1, x.shape[-1])
    # Initialize an empty tensor y to store the output
    y = torch.empty_like(x)
    # Calculate BLOCK_N based on x's element size and ensure it doesn't exceed 64KB
    BLOCK_N = min(65536 // x.element_size(), x.shape[1])
    # Ensure the feature dimension N is not larger than BLOCK_N
    assert x.shape[1] <= BLOCK_N
    # Launch the kernel with the total number of rows M, pointers to x and y, stride, number of columns, eps, and BLOCK_N
    _l2_norm_fwd_1pass_kernel[x.shape[0], BLOCK_N](
        x.data_ptr(), y.data_ptr(), x.stride(0), x.shape[1], eps, BLOCK_N, x.shape[0], tl.ReturnDesc(x.dtype)
    )
    # Return the normalized tensor reshaped to its original dimensions
    return y.reshape(x.shape)
