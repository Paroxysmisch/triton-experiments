# fwd_decay_cumsum
def fwd_decay_cumsum(g, g_o, decay, block_size, grid_size):
    # Initialize a zero vector cum_decay
    cum_decay = zeros(g.shape)
    # Iterate over each block row
    for block_row in range(grid_size):
        # Load a segment from g
        segment = g[block_row * block_size : (block_row + 1) * block_size]
        # Scale the segment by inv_ln2
        segment *= inv_ln2
        # Accumulate into cum_decay
        cum_decay += segment
        # Store the result in g_o
        g_o[block_row * block_size : (block_row + 1) * block_size] = cum_decay

# prepare_qg_kg
def prepare_qg_kg(q, k, g, qg, kg, block_size, grid_size):
    # Iterate over each block row
    for block_row in range(grid_size):
        # Retrieve values from q, k, and g
        q_segment = q[block_row * block_size : (block_row + 1) * block_size]
        k_segment = k[block_row * block_size : (block_row + 1) * block_size]
        g_segment = g[block_row * block_size : (block_row + 1) * block_size]
        # Apply exponential decay and scaling transformations
        qg_segment = exp_decay(q_segment, g_segment) * scaling_factor
        kg_segment = exp_decay(k_segment, g_segment) * scaling_factor
        # Store the results in qg and kg
        qg[block_row * block_size : (block_row + 1) * block_size] = qg_segment
        kg[block_row * block_size : (block_row + 1) * block_size] = kg_segment

# bwd_decay_global_cumsum
def bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, block_size, grid_size):
    # Iterate over each block row
    for block_row in range(grid_size):
        # Load input gradients and tensors
        dq_inner_segment = dq_inner[block_row * block_size : (block_row + 1) * block_size]
        dq_inter_segment = dq_inter[block_row * block_size : (block_row + 1) * block_size]
        dk_inner_segment = dk_inner[block_row * block_size : (block_row + 1) * block_size]
        dk_inter_segment = dk_inter[block_row * block_size : (block_row + 1) * block_size]
        q_segment = q[block_row * block_size : (block_row + 1) * block_size]
        k_segment = k[block_row * block_size : (block_row + 1) * block_size]
        g_segment = g[block_row * block_size : (block_row + 1) * block_size]
        # Compute the gradient of the decay operation
        dg_segment = backward_decay(dq_inner_segment, dq_inter_segment, dk_inner_segment, dk_inter_segment, q_segment, k_segment, g_segment)
        # Accumulate results into dg
        dg[block_row * block_size : (block_row + 1) * block_size] += dg_segment
