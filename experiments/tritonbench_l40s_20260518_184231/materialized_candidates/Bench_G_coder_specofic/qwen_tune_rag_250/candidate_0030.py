tmp11.to(tl.float32)
        tmp16 = tmp10
        tmp17 = tmp16 * tmp15
        tmp18 = tmp3
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20 = tmp17 - tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24 = 1
        tmp25 = tmp5
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp27 = tmp24 * tmp26
        tmp28 = tmp23 / tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.sum(tmp29, 1)[:, None]
        tmp31 = tmp6
        tmp32 = tmp30 / tmp31
        tmp33 = 1
        tmp34 = tmp32 + tmp33
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp11.to(tl.float32)
        tmp37 = tmp3.to(tl.float32)
        tmp38 = tmp36 - tmp37
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp39 * tmp35
        tmp41 = tl.load(in_ptr1 + (x0), eviction_policy="evict_last").to(tl.float32)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp40 * tmp42
        tmp44 = tl.load(in_ptr2 + (x0), eviction_policy="evict_last").to(tl.float32)
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tmp43 + tmp45
        tmp47 = tmp46.to(tl.float32)
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp47, rmask)

def fused_native_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor] = None,
    where: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    x = x.contiguous()
    if out is not None:
        out = out.contiguous()
    else:
        out = torch.empty_like(x)
    if where is not None:
        where = where.contiguous()
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    block_size = triton_heuristics.block_size(x)
    with torch.cuda._DeviceGuard(0):
        triton_red_fused_native_layer_norm_0[(x.shape[0],)](
            out,
            x,
            weight,
            bias,
            out,
            out,
            x.shape[0],
            x.shape[1],
            XBLOCK=1,
            RBLOCK=block_size,
            num_warps=8,
        )
    return out

@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 1024,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    in_out_ptr0,
    in_out_ptr1,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean = tl.sum(tmp2, 1)[:, None]
    tmp3_mean = tmp3_mean / rnumel
    tmp4 = tmp3_mean
    tl.store(in_out_ptr0 + (x0), tmp4, None)
    tmp6 = 1e-05
    tmp7 = tmp6 + tmp4
    tmp8 = libdevice.rsqrt(tmp7)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp8, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp13 = tmp9.to(tl.float32)
        tmp14 = tmp8
        tmp15 = tmp14 * tmp13
        tmp16 = tmp4
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18 = tmp15 - tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22 = 1
        tmp23 = tmp4
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp25 = tmp22 * tmp24
        tmp26 = tmp21 / tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tl.sum(tmp27, 1)[:, None]
        tmp29 = tmp6
        tmp30 = tmp28 / tmp29
        tmp31 = 1
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp13.to(tl.float32)
        tmp35 = tmp4.to(tl.float32)
        tmp36 = tmp34 - tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp37 * tmp33
        tmp39 = tl.load(in_ptr1 + (x0), eviction_policy="evict_last").to(tl.float32)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 * tmp40
        tmp42 = tl.load(in_ptr2 + (x0), eviction_policy="evict_last").to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp41 + tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp45, rmask)

def fused_native_layer_norm_no_welford(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor] = None,
    where: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    x = x.contiguous()
    if out is not None:
        out = out.contiguous()
    else:
        out = torch.empty_like(x)
    if where is not None:
        where = where.contiguous()
    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()
    block_size = triton_heuristics.block_size(x)
    with torch.cuda._DeviceGuard(0):
        triton_red_fused_native_layer_norm_no_welford[(x.shape[0],)](
            out,
            out,
            x,
            weight,
            bias,
            out,
            x.shape[0],
            x.shape[1],
            XBLOCK=1,
            RBLOCK=block_size,
            num
