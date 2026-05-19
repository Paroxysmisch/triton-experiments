import triton
import triton.language as tl

# Kernel for cumulative operations
@triton.jit
def chunk_gated_abc_fwd_kernel_cum(s_ptr, o_ptr, T, S, BT, BS, **meta):
    pid = tl.program_id(axis=0)
    # Calculate block offsets
    row_start = pid * BT
    col_start = 0

    # Define pointers for the block
    s_block_ptr = s_ptr + row_start * S + col_start
    o_block_ptr = o_ptr + row_start * S + col_start

    # Loop over rows in the block
    for t in range(BT):
        row_idx = row_start + t
        if row_idx < T:
            # Load the input
            s_row = tl.load(s_block_ptr + t * S, mask=row_idx < T)
            # Cumulative sum
            if t == 0:
                cum_sum = s_row
            else:
                cum_sum += s_row
            # Store the result
            tl.store(o_block_ptr + t * S, cum_sum, mask=row_idx < T)

# Kernel for gated cumulative sum
@triton.jit
def chunk_gated_abc_fwd_kernel_h(k_ptr, v_ptr, g_ptr, h_ptr, h0_ptr, ht_ptr, T, K, V, BT, BK, BV, **meta):
    pid = tl.program_id(axis=0)
    # Calculate block offsets
    row_start = pid * BT
    col_start = 0

    # Define pointers for the block
    k_block_ptr = k_ptr + row_start * K + col_start
    v_block_ptr = v_ptr + row_start * V + col_start
    g_block_ptr = g_ptr + row_start * V + col_start
    h_block_ptr = h_ptr + row_start * V + col_start

    # Load initial state if needed
    h_state = tl.zeros([BV], dtype=tl.float32)
    if h0_ptr is not None:
        h_state = tl.load(h0_ptr + row_start * V)

    # Loop over rows in the block
    for t in range(BT):
        row_idx = row_start + t
        if row_idx < T:
            # Load inputs
            k_row = tl.load(k_block_ptr + t * K, mask=row_idx < T)
            v_row = tl.load(v_block_ptr + t * V, mask=row_idx < T)
            g_row = tl.load(g_block_ptr + t * V, mask=row_idx < T)

            # Gated cumulative sum
            gated_value = g_row * v_row
            h_state += gated_value

            # Store the result
            tl.store(h_block_ptr + t * V, h_state, mask=row_idx < T)

    # Store final state if needed
    if ht_ptr is not None:
        tl.store(ht_ptr + row_start * V, h_state)

# Wrapper for preparing execution of cumulative kernel
def fwd_pre(s, o, T, S, BT, BS):
    grid = lambda meta: (T + BT - 1) // BT
    chunk_gated_abc_fwd_kernel_cum[grid](s, o, T, S, BT, BS)

# Wrapper for executing gated cumulative sum kernel
def fwd_inner(k, v, g, h, h0, ht, T, K, V, BT, BK, BV):
    grid = lambda meta: (T + BT - 1) // BT
    chunk_gated_abc_fwd_kernel_h[grid](k, v, g, h, h0, ht, T, K, V, BT, BK, BV)
