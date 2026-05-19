",
            device=0,
            target=target,
            device_type=driver.active.utils.get_device_type(device),
            autotune_hints={},
            configs=[],
            compile_options={},
            attrs=attrs,
        )
        kernel = tc.compile(src, output="ttgir", **opts).to_torch()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # allocate output
    y = torch.empty_like(x)

    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix
    grid = (num_programs,)
    kernel[(grid,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y

torch.manual_seed(0)
x = torch.randn(4096, 4096, device="cuda")
y_triton = softmax(x)
