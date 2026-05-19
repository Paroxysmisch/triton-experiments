= occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    # Run kernel
    num_programs = min(num_programs, n_rows)
    grid = (num_programs, 1)
    kernel[(grid, )](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
                     num_warps=num_warps)
    return y
