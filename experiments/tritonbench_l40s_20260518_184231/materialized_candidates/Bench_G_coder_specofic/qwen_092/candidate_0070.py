import triton
import triton.language as tl

def chunk_fwd_h_fn(q, k, v, h, do, initial_state, final_state, BT, block_idx, boundary_check, decay_factor, scale, dtype):
    num_blocks = (q.shape[0] + BT - 1) // BT
    grid = (num_blocks,)
    block = (128,)
    chunk_retention_fwd_kernel_h[grid, block](q, k, v, h, do, initial_state, final_state, BT, block_idx, boundary_check, decay_factor, scale, dtype)
