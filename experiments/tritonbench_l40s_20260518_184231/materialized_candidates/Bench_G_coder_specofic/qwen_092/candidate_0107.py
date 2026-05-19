triton
import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DX, DY, DOUT, OUT,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row = pid
    col = tl.arange(0, BLOCK_N)

    # Load slices of X and Y with boundary conditions handled by masks
    X_slice = tl.load(X + row * BLOCK_N + col)
    Y_slice = tl.load(Y + row * BLOCK_N + col)

    # Compute the derivative of the sigmoid function
    sigmoid = 1 / (1 + tl.exp(-X_slice))
    sigmoid_prime = sigmoid * (1 - sigmoid)

    # Compute gradients using Swish derivative formula for dx and straightforward multiplication for dy
    dx = DOUT * Y_slice * sigmoid_prime
    dy = DOUT * X_slice

    # Store the computed gradients back to memory
    tl.store(DX + row * BLOCK_N + col, dx)
    tl.store(DY + row * BLOCK_N + col, dy)

    # Optionally recompute and store the output tensor
    if RECOMPUTE_OUTPUT:
        out_slice = X_slice * sigmoid
        tl.store(OUT + row * BLOCK_N + col, out_slice)

@triton.jit
def _swiglu_bwd(
    X, Y, DX, DY, DOUT, OUT,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    # Ensure input tensors are contiguous
    X = X.to(tl.float32)
    Y = Y.to(tl.float32)
    DX = DX.to(tl.float32)
    DY = DY.to(tl.float32)
    DOUT = DOUT.to(tl.float32)

    # Reshape for batch dimensions
    batch_size = X.shape[0]
    X = X.reshape(batch_size * BLOCK_N)
    Y = Y.reshape(batch_size * BLOCK_N)
    DX = DX.reshape(batch_size * BLOCK_N)
    DY = DY.reshape(batch_size * BLOCK_N)
    DOUT = DOUT.reshape(batch_size * BLOCK_N)

    # Split xy into x and y
    x = X
    y = Y

    # Set up the execution grid
    num_programs = batch_size
    grid = (num_programs, 1)

    # Launch the kernel
    _swiglu_bwd_kernel[grid](x, y, DX, DY, DOUT, OUT, BLOCK_N, RECOMPUTE_OUTPUT)

    # Adapt outputs based on the recompute_output flag
    if RECOMPUTE_OUTPUT:
        OUT = OUT.reshape(batch_size, BLOCK_N)
    else:
        OUT = None

    return DX, DY, OUT
