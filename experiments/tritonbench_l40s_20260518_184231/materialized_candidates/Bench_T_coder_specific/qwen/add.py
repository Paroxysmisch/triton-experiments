import triton
import torch

def add(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)

    input_shape = input.shape
    other_shape = other.shape

    # Determine the broadcasted shape
    broadcast_shape = tuple(max(dim1, dim2) for dim1, dim2 in zip(input_shape, other_shape))

    # Compute strides
    stride_input_i = input.stride(0)
    stride_input_j = input.stride(1)
    stride_other_i = other.stride(0)
    stride_other_j = other.stride(1)

    # Launch the Triton kernel
    grid = (triton.cdiv(broadcast_shape[0], BLOCK_SIZE_X), triton.cdiv(broadcast_shape[1], BLOCK_SIZE_Y))
    block = (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)

    add_kernel[input_shape[0], block, grid](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        input_shape,
        other_shape,
        stride_input_i,
        stride_input_j,
        stride_other_i,
        stride_other_j,
        input.numel(),
        alpha,
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
    )

    return out
