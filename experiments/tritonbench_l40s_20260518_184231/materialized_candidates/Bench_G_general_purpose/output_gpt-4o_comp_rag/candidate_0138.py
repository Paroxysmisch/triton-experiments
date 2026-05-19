def launch_fwd_decay_cumsum(g, g_o, B, H, T, DK, BT, BK, scale):
    # Compute strides
    s_qk_h = H * T * DK
    s_qk_t = T * DK
    s_qk_d = DK

    # Compute grid dimensions
    grid = (DK // BK, T // BT, B * H)

    # Launch the kernel
    fwd_decay_cumsum[grid](
        g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
        BT=BT, BK=BK, DK=DK
    )
