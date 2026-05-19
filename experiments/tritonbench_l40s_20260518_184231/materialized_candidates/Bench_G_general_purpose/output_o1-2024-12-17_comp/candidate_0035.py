import triton
import triton.language as tl

@triton.jit
def var_len_copy_kernel_triton(
    old_a_start_ptr,  # [num_segments]
    old_a_len_ptr,    # [num_segments]
    old_a_location_ptr,  # [total_old_size]
    new_a_start_ptr,  # [num_segments]
    new_a_location_ptr,  # [total_new_size]
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    old_start = tl.load(old_a_start_ptr + pid)
    length = tl.load(old_a_len_ptr + pid)
    new_start = tl.load(new_a_start_ptr + pid)

    offset = 0
    while offset < length:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (offset + offsets) < length
        src_idx = old_start + offset + offsets
        data = tl.load(old_a_location_ptr + src_idx, mask=mask, other=0)
        dst_idx = new_start + offset + offsets
        tl.store(new_a_location_ptr + dst_idx, data, mask=mask)
        offset += BLOCK_SIZE

def launch_var_len_copy_triton(
    old_a_start,
    old_a_len,
    old_a_location,
    new_a_start,
    new_a_location,
    num_segments
):
    grid = (num_segments,)
    var_len_copy_kernel_triton[grid](
        old_a_start,
        old_a_len,
        old_a_location,
        new_a_start,
        new_a_location,
        BLOCK_SIZE=256
    )
