import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    BLOCK_SIZE: tl.constexpr,
    a_id: tl.constexpr,
    DEBUG: tl.constexpr
):
    # Define the mask to prevent out-of-bound access
    mask = tl.mask(group_size=BLOCK_SIZE)

    # Get the start index and length for the current segment
    old_offset = old_a_start[a_id]
    new_offset = new_a_start[a_id]
    len_a = old_a_len[a_id]

    # Iterate over elements within the block
    for idx in range(0, len_a, BLOCK_SIZE):
        # Load data from the source array
        old_data = tl.load(old_a_location + old_offset + idx, mask=mask)

        # Store data to the destination array
        tl.store(new_a_location + new_offset + idx, old_data, mask=mask)

def launch_var_len_copy_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    BLOCK_SIZE: int,
    num_segments: int
):
    # Prepare the grid size
    grid = lambda meta: (num_segments, meta)

    # Launch the kernel
    var_len_copy_kernel_triton[grid](
        old_a_start, old_a_len, old_a_location,
        new_a_start, new_a_location,
        BLOCK_SIZE,
        a_id=grid.get_group_id().x,
        DEBUG=0
    )
