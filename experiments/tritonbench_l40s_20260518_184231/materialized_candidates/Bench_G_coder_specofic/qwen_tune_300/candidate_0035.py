import triton
import triton.language as tl
import torch

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start, old_a_len, old_a_location, new_a_start, new_a_location, BLOCK_SIZE: tl.constexpr
):
    a_id = tl.program_id(0)
    old_a_start_ptr = tl.make_block_ptr(base=old_a_start, shape=(1,), strides=(1,), offsets=(a_id,), block_shape=(1,))
    new_a_start_ptr = tl.make_block_ptr(base=new_a_start, shape=(1,), strides=(1,), offsets=(a_id,), block_shape=(1,))
    a_start = tl.load(old_a_start_ptr)
    a_len = tl.load(old_a_len + a_id)
    old_offset = 0
    new_offset = 0
    while old_offset < a_len:
        load_mask = old_offset + tl.arange(0, BLOCK_SIZE) < a_len
        old_a_location_ptr = tl.make_block_ptr(base=old_a_location, shape=(1,), strides=(1,), offsets=(a_start + old_offset,), block_shape=(BLOCK_SIZE,), order=(0,))
        new_a_location_ptr = tl.make_block_ptr(base=new_a_location, shape=(1,), strides=(1,), offsets=(new_a_start[a_id] + new_offset,), block_shape=(BLOCK_SIZE,), order=(0,))
        a = tl.load(old_a_location_ptr, boundary_check=(0,), padding_option="zero", mask=load_mask)
        tl.store(new_a_location_ptr, a, boundary_check=(0,), mask=load_mask)
        old_offset += BLOCK_SIZE
        new_offset += BLOCK_SIZE

def launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    num_programs = old_a_len.shape[0]
    grid = (num_programs,)
    var_len_copy_kernel_triton[grid](
        old_a_start,
        old_a_len,
        old_a_location,
        new_a_start,
        new_a_location,
        BLOCK_SIZE,
    )
