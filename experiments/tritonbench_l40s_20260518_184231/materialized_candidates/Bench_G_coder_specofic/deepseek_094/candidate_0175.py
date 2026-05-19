import triton as tl

@triton.jit
def _l2_norm_bwd_kernel(
    x_ptr, dy_ptr, dx_ptr,
    stride_x_row, M, N, eps, BLOCK_N,
    mask_ptr, output_ptr
):
    # Define block indices
    block_idx = tl.program_id(axis=0)
    block_idy = tl.program_id(axis=1)

    # Define block offsets
    offset_x = block_idx * BLOCK_N
    offset_y = block_idy * BLOCK_N

    # Load inputs
    x = tl.load(x_ptr + offset_x, mask_ptr, BLOCK_N, stride_x_row)
    dy = tl.load(dy_ptr + offset_y, mask_ptr, BLOCK_N, 1)

    # Compute variance
    var = tl.sum(x**2) / N

    # Compute reciprocal of standard deviation
    rstd = 1 / tl.sqrt(var + eps)

    # Compute dx
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x

    # Store dx
    tl.store(dx_ptr + offset_x, dx, mask_ptr, BLOCK_N, 1)

def _l2_norm_bwd(x, dy, eps=1e-12, stride_x_row=None):
    # Check for errors
    if x.shape[1] > _MAX_FUSED_SIZE:
        raise ValueError(f"Number of features ({x.shape[1]}) exceeds maximum allowable fused size (_MAX_FUSED_SIZE).")

    # Reshape inputs
    x = x.reshape(-1, x.shape[1])
    dy = dy.reshape(-1, dy.shape[1])

    # Configure strides
    if stride_x_row is None:
        stride_x_row = x.strides[0]

    # Compute block size
    BLOCK_N = min(x.shape[1], _next_power_of_2(x.shape[1]))

    # Create masks
    mask_x = tl.ones((x.shape[0], BLOCK_N), tl.uint8)
    mask_dy = tl.ones((dy.shape[0], BLOCK_N), tl.uint8)

    # Launch kernel
    _l2_norm_bwd_kernel[x.shape[0], 1](
        x.ctypes.data, dy.ctypes.data, x.ctypes.data,
        stride_x_row, x.shape[0], x.shape[1], eps, BLOCK_N,
        mask_x.ctypes.data, mask_dy.ctypes.data
    )

    # Reshape outputs
    dx = x.reshape(dy.shape)

    return dx
