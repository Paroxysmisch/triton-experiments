=0)
    three_d_idx = tl.program_id(axis=1)

    philox_offsets = tl.arange(0, block_size)

    # Get the seed for the current row.
    row_seed = tl.load(seed_ptr + row_idx * seed_row_stride)
    if three_d_idx > 0:
        row_seed ^= three_d_idx

    # Generate random numbers in [0, 1).
    out1, out2, out3, out4 = tl.rand4x(row_seed, philox_offsets)

    output_row_start_ptr = (out_ptr + row_idx * out_row_stride +
                            three_d_idx * out_3d_stride)
    out1_offsets = philox_offsets
    tl.store(output_row_start_ptr + out1_offsets,
             out1,
             mask=out1_offsets < n_cols)
    if n_slices > 1:
        out2_offsets = tl.arange(block_size, block_size * 2)
        tl.store(output_row_start_ptr + out2_offsets,
                 out2,
                 mask=out2_offsets < n_cols)
    if n_slices > 2:
        out3_offsets = tl.arange(block_size * 2, block_size * 3)
        tl.store(output_row_start_ptr + out3_offsets,
                 out3,
                 mask=out3_offsets < n_cols)
    if n_slices > 3:
        out4_offsets = tl.arange(block_size * 3, block_size * 4)
        tl.store(output_row_start_ptr + out4_offsets,
                 out4,
                 mask=out4_offsets < n_cols)
