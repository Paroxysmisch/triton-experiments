seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen +
                  rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen +
                  rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (
                rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (
                rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen +
                     rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen)
                 & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch_size, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_size = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    rotary_dim = cos.shape[-1] * 2
    assert (
        rotary_dim <= headdim
    ), "rotary_dim must be <= headdim, but found {} > {}".format(rotary_dim, headdim)
    assert headdim <= 256, "Only support headdim <= 256"
    assert rotary_dim % 2 == 0
    assert seqlen <= 2048, "Only support seqlen <= 2048"
    if not is_varlen:
        seqlen_ro = min(seqlen, 2048)
        cos, sin = cos.squeeze(0), sin.squeeze(0)
    else:
        seqlen_ro = min(max_seqlen, 2048)
    x = x.contiguous()
    if not inplace:
        output = torch.empty_like(x)
    else:
        output = x
    assert (
        cos.stride()[-1] == 1
        and sin.stride()[-1] == 1
        and x.stride()[-1] == 1
        and output.stride()[-1] == 1
    ), "Strides must be contiguous for last dimension"
    grid = lambda META: (
        triton.cdiv(seqlen, META["BLOCK_M"]),
        batch_size,
        nheads,
    )
    rotary_kernel[grid](
        output,
        x,
        cos,
        sin,
        cu_seqlens,
        seqlen_offsets,
        seqlen,
        nheads,
        rotary_dim,
        seqlen_ro,
        max_seqlen if is_varlen else 0,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_K=triton.next_power_of_2(rotary_dim),
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=is_varlen,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
    )
    return output
