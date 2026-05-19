= NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    # launch kernel
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y
