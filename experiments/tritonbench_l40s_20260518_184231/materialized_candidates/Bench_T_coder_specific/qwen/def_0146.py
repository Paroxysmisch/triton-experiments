import triton
import triton.language as tl

@triton.jit
def elu_linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    X_shape, W_shape, B_shape, Y_shape,
    alpha, inplace,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr):
    
    # Get the indices for the current block
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Calculate the base offsets
    X_base = row * X_shape[1]
    W_base = col * W_shape[0]

    # Initialize output values
    y = tl.zeros((BLOCK_SIZE_Y,), dtype=tl.float32)

    # Load input data
    for i in range(BLOCK_SIZE_X):
        x = tl.load(X_ptr + X_base + i)

        # Perform linear transformation
        wx = tl.dot(x, tl.load(W_ptr + W_base + i))

        # Add bias if it exists
        if B_ptr is not None:
            wx += tl.load(B_ptr + i)

        # Apply ELU activation
        if inplace:
            y[i] = tl.where(wx > 0, wx, alpha * (tl.exp(wx) - 1))
        else:
            y[i] = tl.select(wx > 0, wx, alpha * (tl.exp(wx) - 1))

    # Store the result back to memory
    tl.store(Y_ptr + row * Y_shape[1], y)

# Launch the kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 512}, num_stages=1, num_warps=4),
    ],
    key=['X_shape', 'W_shape', 'Y_shape']
)
def elu_linear_forward(X, W, B, Y, alpha, inplace):
    X_shape = X.shape
    W_shape = W.shape
    B_shape = B.shape if B is not None else None
    Y_shape = Y.shape

    grid = (
        triton.cdiv(X_shape[0], BLOCK_SIZE_X),
        triton.cdiv(Y_shape[1], BLOCK_SIZE_Y)
    )

    elu_linear_kernel[grid](
        X.data_ptr(), W.data_ptr(), B.data_ptr() if B is not None else None, Y.data_ptr(),
        X_shape, W_shape, B_shape, Y_shape,
        alpha, inplace,
        BLOCK_SIZE_X, BLOCK_SIZE_Y
    )
