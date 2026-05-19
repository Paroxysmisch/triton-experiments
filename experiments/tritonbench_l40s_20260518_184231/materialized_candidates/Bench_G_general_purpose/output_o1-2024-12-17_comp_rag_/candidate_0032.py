import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)

    for i in range(0, length, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < length
        v = tl.load(old_a_location + old_start + offset, mask=mask)
        tl.store(new_a_location + new_start + offset, v, mask=mask)

def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)
    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE
    )
