@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    o_ptr, s_ptr, B, H, T, BT,
    num_warps=triton.num_warps,
    num_stages=1,
    num_threads=triton.num_threads_per_warp,
    num_warps_per_block=1,
    num_blocks_x=1,
    **kwargs
):
    grid = tc.cg.get_grid(num_blocks_x=num_blocks_x, num_threads_per_block=num_warps_per_block * num_stages * num_threads)
    lane_id = tc.cg.get_lane_id()
    warp_id = tc.cg.get_warp_id()
    block_id = tc.cg.get_block_id()
    block_size = num_warps_per_block * num_stages * num_threads
    b = block_id // H
    h = block_id % H
    offset = b * H * T + h * T
    s_ptrs = [s_ptr + offset + i * T for i in range(num_warps_per_block)]
    o_ptrs = [o_ptr + offset + i * T for i in range(num_warps_per_block)]
    for t in trange(T - 1, -1, -1):
        v = tc.cg.load(s_ptrs[warp_id], t)
        if t < BT:
            v += tc.cg.load(s_ptrs[(warp_id + 1) % num_warps_per_block], t)
        tc.cg.store(o_ptrs[warp_id], t, v)

def chunk_global_reversed_cumsum_scalar(
    o, s, B, H, T, BT,
    num_warps=128,
    num_stages=1,
    num_threads=32,
    num_warps_per_block=1,
    num_blocks_x=-1,
    **kwargs
):
    if num_blocks_x == -1:
        num_blocks_x = B * H
    s_ptr = s.data_ptr()
    o_ptr = o.data_ptr()
    chunk_global_reversed_cumsum_scalar_kernel[num_blocks_x, num_warps_per_block](
        o_ptr, s_ptr, B, H, T, BT,
        num_warps=num_warps,
        num_stages=num_stages,
        num_threads=num_threads,
        **kwargs
    )
