import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start_ptr,
    old_a_len_ptr,
    old_a_location_ptr,
    new_a_start_ptr,
    new_a_location_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    a_id = tl.program_id(0)
    
    old_start = tl.load(old_a_start_ptr + a_id)
    new_start = tl.load(new_a_start_ptr + a_id)
    length = tl.load(old_a_len_ptr + a_id)
    
    for offset in range(0, length, BLOCK_SIZE):
        positions = offset + tl.arange(0, BLOCK_SIZE)
        mask = positions < length
        
        old_pos = old_start + positions
        new_pos = new_start + positions
        
        data = tl.load(old_a_location_ptr + old_pos, mask=mask)
        tl.store(new_a_location_ptr + new_pos, data, mask=mask)

def launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location):
    assert old_a_start.is_cuda and old_a_len.is_cuda and old_a_location.is_cuda
    assert new_a_start.is_cuda and new_a_location.is_cuda
    
    n_segments = old_a_start.shape[0]
    assert old_a_len.shape == (n_segments,) and new_a_start.shape == (n_segments,)
    
    grid = (n_segments,)
    BLOCK_SIZE = 256
    
    var_len_copy_kernel_triton[grid](
        old_a_start, old_a_len, old_a_location,
        new_a_start, new_a_location,
        BLOCK_SIZE=BLOCK_SIZE
    )
