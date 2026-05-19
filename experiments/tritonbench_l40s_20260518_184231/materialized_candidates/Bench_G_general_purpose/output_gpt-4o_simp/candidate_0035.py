import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def var_len_copy_kernel_triton(
    src_ptr, dst_ptr, src_indices_ptr, dst_indices_ptr, lengths_ptr, 
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID for this kernel instance
    pid = tl.program_id(axis=0)

    # Load the starting index and length for the current segment
    src_index = tl.load(src_indices_ptr + pid)
    dst_index = tl.load(dst_indices_ptr + pid)
    length = tl.load(lengths_ptr + pid)

    # Create a range for the current block
    offsets = tl.arange(0, BLOCK_SIZE)

    # Calculate the effective indices for the source and destination
    src_offsets = src_index + offsets
    dst_offsets = dst_index + offsets

    # Load from source and store to destination within the segment length
    mask = offsets < length
    src_vals = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + dst_offsets, src_vals, mask=mask)

# Wrapper function to launch the Triton kernel
def launch_var_len_copy_triton(src, dst, src_indices, dst_indices, lengths, block_size=128):
    assert len(src_indices) == len(dst_indices) == len(lengths), "Segment arrays must be of the same length"

    # Determine the number of segments
    num_segments = len(src_indices)

    # Allocate memory on the device
    src_ptr = triton.device_pointer(src)
    dst_ptr = triton.device_pointer(dst)
    src_indices_ptr = triton.device_pointer(src_indices)
    dst_indices_ptr = triton.device_pointer(dst_indices)
    lengths_ptr = triton.device_pointer(lengths)

    # Launch the Triton kernel
    grid = (num_segments,)
    var_len_copy_kernel_triton[grid](
        src_ptr, dst_ptr, src_indices_ptr, dst_indices_ptr, lengths_ptr, 
        BLOCK_SIZE=block_size
    )
