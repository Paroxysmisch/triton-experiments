import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr,  # input tensor
    o_ptr,  # output tensor
    B, H, T,  # batch size, head count, time steps
    BT,  # block size in time steps
    stride_BH, stride_B, stride_T,  # strides for the input tensor
    stride_OH, stride_O, stride_OT,  # strides for the output tensor
    BLOCK_SIZE: tl.constexpr,  # block size in threads
):
    # Compute the indices
    b = tl.program_id(0)  # batch index
    h = tl.program_id(1)  # head index
    t = tl.program_id(2)  # time index

    # Initialize the running sum
    running_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the time steps in chunks
    for chunk in range(0, T, BT):
        # Load a block of data
        block_ptr = tl.make_block_ptr(
            base_ptr=s_ptr,
            index=(b, h, chunk),
            shape=(B, H, T),
            strides=(stride_BH, stride_B, stride_T),
            block_shape=(BLOCK_SIZE, 1, BT),
            order=(0, 1, 2),
        )
        block_data = tl.load(block_ptr)

        # Perform cumulative sum
        block_cumsum = tl.cumsum(block_data, axis=2)

        # Update the running sum
        running_sum = running_sum + block_cumsum[:, :, -1]

        # Store the result
        result_ptr = tl.make_block_ptr(
            base_ptr=o_ptr,
            index=(b, h, chunk),
            shape=(B, H, T),
            strides=(stride_OH, stride_O, stride_OT),
            block_shape=(BLOCK_SIZE, 1, BT),
            order=(0, 1, 2),
        )
        tl.store(result_ptr, block_cumsum)

    # Store the running sum as the final result
    final_result_ptr = tl.make_block_ptr(
        base_ptr=o_ptr,
        index=(b, h, T),
        shape=(B, H, T),
        strides=(stride_OH, stride_O, stride_OT),
        block_shape=(BLOCK_SIZE, 1, 1),
        order=(0, 1, 2),
    )
    tl.store(final_result_ptr, running_sum[:, :, None])

@triton.jit
def chunk_global_cumsum_scalar(s, dtype=None):
    if dtype is None:
        dtype = s.dtype

    B, H, T = s.shape
    BT = 128  # block size in time steps

    # Initialize output tensor
    z = tl.zeros((B, H, T), dtype=dtype)

    # Grid configuration
    grid = (B, H, (T + BT - 1) // BT)

    # Launch the kernel
    chunk_global_cumsum_scalar_kernel[grid](s, z, B, H, T, BT, s.stride(0), s.stride(1), s.stride(2), z.stride(0), z.stride(1), z.stride(2), BLOCK_SIZE=128)

    return z
