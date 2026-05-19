import triton
import triton.language as tl

@triton.jit
def fwd_decay_cumsum(g_ptr, g_o_ptr, decay_ptr, BT: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize indices
    row = tl.program_id(0)
    col = tl.program_id(1)
    offset = row * BLOCK_SIZE + col

    # Load input tensor g
    g = tl.load(g_ptr + offset)

    # Initialize cum_decay with zero
    cum_decay = 0.0

    # Iterate over the defined number of block rows
    for i in range(BT):
        # Scale by inv_ln2
        scaled_g = g * tl.exp(-decay_ptr[i])

        # Accumulate into cum_decay
        cum_decay += scaled_g

    # Store the result in g_o
    tl.store(g_o_ptr + offset, cum_decay)

# Launch function for fwd_decay_cumsum
def launch_fwd_decay_cumsum(g, g_o, decay, BT, BLOCK_SIZE):
    grid = (g.shape[0] // BT, g.shape[1] // BLOCK_SIZE)
    block = (BLOCK_SIZE, 1)
    triton.launch_kernel[fwd_decay_cumsum](g, g_o, decay, BT, BLOCK_SIZE, grid=grid, block=block)
