import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start_ptr, old_a_len_ptr, old_a_location_ptr,
    new_a_start_ptr, new_a_location_ptr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    a_id = tl.program_id(0)

    # Load the start index and length for this segment
    old_start = tl.load(old_a_start_ptr + a_id)
    length = tl.load(old_a_len_ptr + a_id)

    # Calculate the number of blocks needed for this segment
    num_blocks = tl.cdiv(length, BLOCK_SIZE)

    # Iterate over blocks
    for block in range(num_blocks):
        # Calculate offsets
        old_offset = old_start + block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        new_offset = tl.load(new_a_start_ptr + a_id) + block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Create a mask for valid elements in this block
        mask = old_offset < (old_start + length)

        # Load data from old location
        old_data = tl.load(old_a_location_ptr + old_offset, mask=mask)

        # Store data to new location
        tl.store(new_a_location_ptr + new_offset, old_data, mask=mask)

def launch_var_len_copy_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    num_segments
):
    BLOCK_SIZE = 256

    # Launch the kernel
    var_len_copy_kernel_triton[(num_segments,)](
        old_a_start, old_a_len, old_a_location,
        new_a_start, new_a_location,
        BLOCK_SIZE=BLOCK_SIZE
    )
