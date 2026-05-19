import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    X_ptr, Y_ptr, Z_ptr,
    X_shape, Y_shape, Z_shape,
    stride_xz, stride_yz,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Perform elementwise multiplication of two tensors.

    Args:
        X_ptr (Pointer): Pointer to the input tensor X.
        Y_ptr (Pointer): Pointer to the input tensor Y.
        Z_ptr (Pointer): Pointer to the output tensor Z.
        X_shape (tuple): Shape of tensor X.
        Y_shape (tuple): Shape of tensor Y.
        Z_shape (tuple): Shape of tensor Z.
        stride_xz (int): Stride from X to Z.
        stride_yz (int): Stride from Y to Z.
        BLOCK_SIZE (int): Block size for parallel computation.
    """
    coords = tl.program_id(axis=0)
    coords_in_block = tl.arange(0, BLOCK_SIZE)
    coords = coords * BLOCK_SIZE + coords_in_block

    # Broadcast coordinates to match shapes
    x_coords = coords % X_shape[0]
    y_coords = coords % Y_shape[0]

    # Load values from X and Y
    x_val = tl.load(X_ptr + x_coords, mask=x_coords < X_shape[0])
    y_val = tl.load(Y_ptr + y_coords, mask=y_coords < Y_shape[0])

    # Compute multiplication
    z_val = x_val * y_val

    # Store result in Z
    tl.store(Z_ptr + coords, z_val, mask=(coords < Z_shape[0]))
