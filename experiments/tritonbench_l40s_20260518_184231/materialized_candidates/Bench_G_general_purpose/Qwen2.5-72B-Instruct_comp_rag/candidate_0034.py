import triton
import triton.language as tl

# Triton kernel function
@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location, BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    a_id = tl.program_id(0)
    
    # Load the length, old start index, and new start index for the current segment
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    
    # Create offset arrays for the block
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    
    # Loop over the segment in blocks
    for i in range(0, length, BLOCK_SIZE):
        # Load the data from the old location with a mask to prevent out-of-bound accesses
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length - i)
        
        # Store the data to the new location with a mask to prevent out-of-bound accesses
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length - i)

# Wrapper function to launch the kernel
def launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)
    
    # Launch the kernel with the specified grid size and block size
    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_a_location, new_a_start, new_a_location, BLOCK_SIZE
    )
