import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start_ptr, old_a_len_ptr, old_a_location_ptr,
    new_a_start_ptr, new_a_location_ptr,
    BLOCK_SIZE: tl.constexpr,
    a_id: tl.program_id
):
    # Compute the starting index and length for the current segment
    old_start = tl.load(old_a_start_ptr + a_id)
    old_len = tl.load(old_a_len_ptr + a_id)
    new_start = tl.load(new_a_start_ptr + a_id)
    
    # Calculate the offsets for this block
    old_offset = old_start + tl.arange(0, BLOCK_SIZE)
    new_offset = new_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle out-of-bound accesses
    mask = old_offset < (old_start + old_len)
    
    # Load from old array and store into new array
    data = tl.load(old_a_location_ptr + old_offset, mask=mask, other=0)
    tl.store(new_a_location_ptr + new_offset, data, mask=mask)

def launch_var_len_copy_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    num_segments, BLOCK_SIZE=256
):
    # Convert inputs to Triton pointers
    old_a_start_ptr = old_a_start.ctypes.data
    old_a_len_ptr = old_a_len.ctypes.data
    old_a_location_ptr = old_a_location.ctypes.data
    new_a_start_ptr = new_a_start.ctypes.data
    new_a_location_ptr = new_a_location.ctypes.data
    
    # Launch the Triton kernel
    grid = (num_segments,)
    var_len_copy_kernel_triton[grid](
        old_a_start_ptr, old_a_len_ptr, old_a_location_ptr,
        new_a_start_ptr, new_a_location_ptr,
        BLOCK_SIZE=BLOCK_SIZE
    )
