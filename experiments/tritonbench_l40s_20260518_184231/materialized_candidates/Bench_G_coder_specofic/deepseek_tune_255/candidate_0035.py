import triton
import triton.language as tl
import torch

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    BLOCK_SIZE: tl.constexpr
):
    a_id = tl.program_id(0)
    # read info
    old_start = tl.load(old_a_start + a_id)
    old_len = tl.load(old_a_len + a_id)
    # looped copy
    for i in range(0, old_len):
        old_offset = (old_start + i)
        new_offset = (old_start + i)
        # masks
        old_mask = (i + tl.arange(0, BLOCK_SIZE)) < old_len
        new_mask = old_mask
        # copy
        old = tl.load(old_a_location + old_offset, mask=old_mask)
        tl.store(new_a_location + new_offset, old, mask=new_mask)

def launch_var_len_copy_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    num_segments,
    BLOCK_SIZE
):
    grid = (num_segments,)
    var_len_copy_kernel_triton[grid](
        old_a_start, old_a_len, old_a_location,
        new_a_start, new_a_location,
        BLOCK_SIZE=BLOCK_SIZE
    )

def var_len_copy_triton(
    old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location,
    num_segments
):
    BLOCK_SIZE = 256
    launch_var_len_copy_triton(
        old_a_start, old_a_len, old_a_location,
        new_a_start, new_a_location,
        num_segments,
        BLOCK_SIZE
    )
