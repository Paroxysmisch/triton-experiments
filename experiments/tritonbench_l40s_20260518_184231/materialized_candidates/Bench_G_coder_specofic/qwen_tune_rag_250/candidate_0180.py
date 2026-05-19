.stride(-1) == 1
    assert dx.stride(-1) == 1
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _l2_norm_bwd_kernel[(M,)](
            x,
            dy,
            dx,
            x.stride(0),
            N,
            eps,
            BLOCK_N,
        )
    return dx.reshape(x_shape_og)
