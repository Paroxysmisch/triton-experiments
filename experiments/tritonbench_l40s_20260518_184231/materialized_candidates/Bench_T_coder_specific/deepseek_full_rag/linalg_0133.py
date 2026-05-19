s, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = min(n_rows, BLOCK_SIZE * occupancy * num_warps * num_stages)
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    # Run kernel
    grid = lambda meta: (num_programs, )
    kernel[grid](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
                 num_warps=num_warps)
    return y
