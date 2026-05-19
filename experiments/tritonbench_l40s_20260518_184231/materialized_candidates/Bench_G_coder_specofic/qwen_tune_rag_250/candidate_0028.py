rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # store
        OUT = OUT + (rm[:, None] * stride_out_seqlen +
                     rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen)
                 & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
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
):
    """
    Apply rotary on the input tensor `x`
    Args:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: sequence length offset(s) for the batch elements.
            It can be either an integer or a tensor with batch size.
        cu_seqlens: (batch + 1) if variable seqlen else None.
        max_seqlen: int, the max seqlen for variable seqlen.
        interleaved: bool, whether the input is interleaved.
        inplace: bool, whether to perform the operation inplace.
        conjugate: bool, whether to conjugate the sin component.
    Return:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "max_seqlen must be provided if cu_seqlens is not None"
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "headdim must be <= 256"
    assert seqlen <= 2048, "seqlen must be <= 2048"
    assert cos.shape == sin.shape
    assert cos.shape[-1] == headdim // 2
    if not is_varlen:
        assert (
            x.stride()[-2:] == (1, rotary_dim)
            or x.stride()[-2:] == (rotary_dim, 1)
            or (not interleaved and x.stride()[-2:] == (headdim, 1))
        ), "Invalid strides for x"
    else:
        assert x.stride()[-2:] == (headdim, 1), "Invalid strides for x"
    assert cos.stride() == (1, rotary_dim // 2) or cos.stride() == (
        rotary_dim // 2, 1
    ), "Invalid strides for cos"
    assert sin.stride() == (1, rotary_dim // 2) or sin.stride() == (
        rotary_dim // 2, 1
    ), "Invalid strides for sin"
    if not inplace:
        # Need to make a copy of x if not inplace
        x = x.contiguous()
        if is_varlen:
            out = torch.empty_like(x, dtype=torch.float32)
        else:
            out = torch.empty_like(x)
    else:
        out = x
    # prepare kernel launch params
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.stride() == (1,)
        is_seqlen_offsets_tensor = True
    else:
        seqlen_offsets = seqlen_offsets.to(torch.int32)
        is_seqlen_offsets_tensor = False
    # work out num_warps
    num_warps = 4
    if rotary_dim >= 128:
        num_warps = 8
    if rotary_dim >= 256:
        num_warps = 16
    # enqueue kernel
    grid = lambda META: (
        triton.cdiv(seqlen, META["BLOCK_M"]),
        batch,
        nheads,
    )
    rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        cu_seqlens,
        seqlen_offsets,
        seqlen,
        nheads,
        rotary_dim,
        min(seqlen, rotary_dim * 2),
        max_seqlen if is_varlen else 1,
        seqlen if is_varlen else 1,
        batch,
        seqlen,
        nheads,
        headdim,
        1,
        seqlen,
        nheads,
        headdim,
        BLOCK_K=triton.next_power_of_2(rotary_dim),
        IS_SEQLEN_OFFSETS_TENSOR=is_seqlen_offsets_tensor,
        IS_VARLEN=is_varlen,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        num_warps=num_warps,
    )
    return out
