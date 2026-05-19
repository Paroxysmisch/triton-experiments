import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr, o_ptr, B, H, T, BT,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program index for batch and head dimensions
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Calculate the offset for the current batch and head
    batch_offset = batch_idx * H * T
    head_offset = head_idx * T
    base_offset = batch_offset + head_offset

    # Initialize running sum
    running_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over chunks along the Time dimension
    for t_start in range(0, T, BLOCK_SIZE):
        # Compute the offset for the current chunk
        offset = base_offset + t_start

        # Create block pointer and load the data
        s_block_ptr = tl.make_block_ptr(s_ptr + offset, shape=[B, H, T], strides=[H * T, T, 1], offsets=[batch_idx, head_idx, t_start], block_shape=[1, 1, BLOCK_SIZE])
        s_block = tl.load(s_block_ptr)

        # Perform cumulative sum on the block
        cumsum_block = tl.cumsum(s_block, axis=0) + running_sum

        # Store the cumulative sum result in the output tensor
        o_block_ptr = tl.make_block_ptr(o_ptr + offset, shape=[B, H, T], strides=[H * T, T, 1], offsets=[batch_idx, head_idx, t_start], block_shape=[1, 1, BLOCK_SIZE])
        tl.store(o_block_ptr, cumsum_block)

        # Update running sum for the next chunk
        running_sum = tl.sum(cumsum_block, axis=0)

# Define the wrapper function
def chunk_global_cumsum_scalar(s: torch.Tensor, dtype=None, BT=1024):
    # Ensure dtype is set
    if dtype is None:
        dtype = s.dtype

    # Get the dimensions of the input tensor
    B, H, T = s.shape

    # Initialize the output tensor
    z = torch.empty_like(s, dtype=dtype)

    # Determine grid size
    grid = (B, H)

    # Launch the kernel
    triton.launch(
        chunk_global_cumsum_scalar_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        args=[s, z, B, H, T, BT],
        BLOCK_SIZE=BT
    )

    return z

# Example usage
s = torch.randn(2, 3, 1024, dtype=torch.float32, device='cuda')
z = chunk_global_cumsum_scalar(s)
