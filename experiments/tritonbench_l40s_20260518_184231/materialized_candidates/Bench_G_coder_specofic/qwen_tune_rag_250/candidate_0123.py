<= o_i[None, :], 1., 0.)

    b_ds = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_dz = tl.make_block_ptr(dz + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_dz = tl.load(p_dz, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_ds[None, :] + tl.dot(m_s, b_dz, allow_tf32=False)
        tl.store(p_ds, b_c.to(p_ds.dtype.element_ty), boundary_check=(0, 1))

        if i_t >= 0:
            b_ds += tl.sum(b_dz, 0)


def chunk_cumsum_fwd(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32

    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    chunk_cumsum_fwd_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z


def chunk_cumsum_bwd(
    dz: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    B, H, T, S = dz.shape
    BS = 32

    dtype = dtype or dz.dtype
    grid = (triton.cdiv(S, BS), B * H)
    ds = torch.empty_like(dz, dtype=dtype)
    chunk_cumsum_bwd_kernel[grid](
        ds, dz,
        ds.stride(1), ds.stride(2), ds.stride(3),
        T=T, S=S, BS=BS
    )
    return ds
