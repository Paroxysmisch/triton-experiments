@triton.jit
def conv2d_kernel(
    x,
    weight,
    output,
    stride,
    padding,
    dilation,
    groups,
    x_shape,
    weight_shape,
    output_shape,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    i = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    j = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    k = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)

    i = i * stride - padding
    j = j * stride - padding

    o_i = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    o_j = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    o_i = tl.min(o_i, x_shape[1] - 1)
    o_j = tl.min(o_j, x_shape[2] - 1)

    x_offset = pid_z * BLOCK_SIZE_Z * x_shape[1] * x_shape[2] + i * x_shape[2] + j
    w_offset = pid_z * BLOCK_SIZE_Z * weight_shape[1] * weight_shape[2] + k * weight_shape[2] + 0

    x_val = tl.load(x + x_offset, mask=i < x_shape[1] and j < x_shape[2], other=0.0)
    w_val = tl.load(weight + w_offset, mask=k < weight_shape[0], other=0.0)

    acc = tl.dot(x_val, w_val)
    tl.atomic_add(output + o_i * output_shape[2] + o_j, acc)
