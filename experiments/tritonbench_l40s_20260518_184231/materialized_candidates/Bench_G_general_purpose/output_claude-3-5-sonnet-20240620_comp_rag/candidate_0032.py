import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start,    # *Pointer* to array of start indices for old segments
    old_a_len,      # *Pointer* to array of segment lengths
    old_a_location, # *Pointer* to source data array
    new_a_start,    # *Pointer* to array of start indices for new segments
    new_a_location, # *Pointer* to destination data array
    BLOCK_SIZE: tl.constexpr,  # Block size as compile-time constant
):
    # Get the unique program instance ID
    a_id = tl.program_id(0)
    
    # Load the length and start positions for this segment
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    
    # Create offset arrays for vectorized operations
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    
    # Process the segment in blocks
    for i in range(0, length, BLOCK_SIZE):
        # Create masks for bounds checking
        mask = old_offset < (length - i)
        
        # Load data from source with bounds checking
        v = tl.load(old_a_location + old_start + i + old_offset, mask=mask)
        
        # Store data to destination with bounds checking
        tl.store(new_a_location + new_start + i + new_offset, v, mask=mask)

def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    """
    Launch the variable length copy kernel
    
    Args:
        old_a_start: Start indices for old segments
        old_a_len: Lengths of segments
        old_location: Source data array
        new_a_start: Start indices for new segments
        new_a_location: Destination data array
    """
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)  # One thread block per segment
    
    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, 
        new_a_start, new_a_location, 
        BLOCK_SIZE
    )
