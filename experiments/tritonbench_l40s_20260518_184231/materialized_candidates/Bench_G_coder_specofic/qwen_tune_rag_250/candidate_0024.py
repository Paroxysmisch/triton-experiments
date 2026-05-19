=dtype)
    s = s.contiguous()
    z = z.contiguous()

    chunk_cumsum_fwd_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T, S, BS=BS
    )
    return z


def chunk_cumsum_bwd(
    dz: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = dz.shape
    BS = 32

    dtype = dtype or dz.dtype
    grid = (triton.cdiv(S, BS), B * H)
    ds = torch.empty_like(dz, dtype=dtype)
    ds = ds.contiguous()
    dz = dz.contiguous()

    chunk_cumsum_bwd_kernel[grid](
        ds, dz,
        ds.stride(1), ds.stride(2), ds.stride(3),
        T, S, BS=BS
    )
    return ds
