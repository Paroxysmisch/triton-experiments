import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    # Get the unique program identifier to determine which segment to process
    a_id = tl.program_id(0)
    
    # Load the length and start indices for the current segment
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    
    # Define offsets within the block
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    
    # Loop to copy data in chunks of BLOCK_SIZE
    for i in range(0, length, BLOCK_SIZE):
        # Calculate mask to prevent out-of-bound accesses
        mask = old_offset < length - i
        
        # Load data from old location with masking
        v = tl.load(old_a_location + old_start + i + old_offset, mask=mask)
        
        # Store data to new location with masking
        tl.store(new_a_location + new_start + i + new_offset, v, mask=mask)

def launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)  # Define grid size based on the number of segments

    # Launch the kernel with the specified grid size and block size
    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_a_location, new_a_start, new_a_location, BLOCK_SIZE
    )
