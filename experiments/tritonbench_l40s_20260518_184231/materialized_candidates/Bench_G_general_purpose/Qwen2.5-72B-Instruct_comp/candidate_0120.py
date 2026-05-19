import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr,  # Pointer to input tensor
    o_ptr,  # Pointer to output tensor
    B,      # Batch size
    H,      # Head count
    T,      # Time steps
    BT,     # Block size for Time dimension
    BLOCK_SIZE_BT: tl.constexpr
):
    # Get the program ID for batch and head
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Initialize running sum for each block
    running_sum = tl.zeros((BLOCK_SIZE_BT,), dtype=tl.float32)

    # Loop over chunks of the time dimension
    for chunk_start in range(0, T, BT):
        chunk_end = min(chunk_start + BT, T)
        chunk_size = chunk_end - chunk_start

        # Create block pointers for input and output
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=(B, H, T),
            strides=(H * T, T, 1),
            offsets=(pid_b, pid_h, chunk_start),
            block_shape=(1, 1, BLOCK_SIZE_BT),
            order=(2, 1, 0)
        )
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr,
            shape=(B, H, T),
            strides=(H * T, T, 1),
            offsets=(pid_b, pid_h, chunk_start),
            block_shape=(1, 1, BLOCK_SIZE_BT),
            order=(2, 1, 0)
        )

        # Load the block of data
        s_block = tl.load(s_block_ptr, mask=chunk_start + tl.arange(0, BLOCK_SIZE_BT) < T)

        # Perform cumulative sum within the block
        cumsum_block = tl.cumsum(s_block, axis=2)

        # Add the running sum to the cumulative sum
        cumsum_block += running_sum

        # Store the result in the output tensor
        tl.store(o_block_ptr, cumsum_block, mask=chunk_start + tl.arange(0, BLOCK_SIZE_BT) < T)

        # Update the running sum
        running_sum = cumsum_block[chunk_size - 1]

### Wrapper Function: `chunk_global_cumsum_scalar`
