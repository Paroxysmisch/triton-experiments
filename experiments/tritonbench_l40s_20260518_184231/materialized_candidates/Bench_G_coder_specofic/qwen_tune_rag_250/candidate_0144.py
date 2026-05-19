}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 8}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 16}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 32}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 64}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])
        ),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_bwd_kernel(
    ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr, ddt_ptr, ddt_bias_ptr,
    dt_min, dt_max, stride_ddA_batch, stride_ddA_chunk, stride_ddA_head,
    stride_ddA_csize, stride_ddt_out_batch, stride_ddt_out_chunk, stride_ddt_out_head,
    stride_ddt_out_csize, stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head, stride_ddt_batch, stride_ddt_seqlen, stride_ddt_head,
    stride_ddt_bias_head, DT_SOFTPLUS: tl.constexpr, HAS_DDT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += (
        pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk + (chunk_size - 1)
        * stride_ddt_out_csize
    )
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk + (
        chunk_size - 1
    ) * stride_ddA_csize
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_seqlen

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = (
        ddt_out_ptr + (offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize)
    )
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    ddA = tl.load(
        ddA_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    ddt_out = tl.load(
        ddt_out_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out * dt
    offs_cm = tl.arange(0, BLOCK_SIZE_CHUNK)[None, :] + (chunk_size - 1)[:, None]
    ddt_m = tl.load(
        ddt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_cm < chunk_size),
        other=0.0,
    ).to(tl.float32)
    ddt = ddt + ddt_m
    if HAS_DDT_BIAS:
        ddt_bias = tl.load(
            ddt_bias_ptr + offs_h * stride_ddt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        ddt += ddt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = tl.load(
            dt_ptr + offs_h * stride_dt_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        ddt = ddt * tl.sigmoid(dt_presoftplus)
    ddt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0
    )
    tl.store(
        ddt_ptrs,
        ddt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    has_heads = dt.dim() == 3
    if dt.dim() == 2:
        dt = dt.unsqueeze(2)
    elif dt.dim() > 3:
        raise ValueError(f"dt must be 2 or 3 dimenstional, got {dt.dim()}")
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if A.dim() == 1:
        A = A.unsqueeze(0)
    elif A.dim() > 2:
        raise ValueError(f"A must be 1 or 2 dimenstional, got {A.dim()}")
    batch, seqlen, nheads = dt.shape
    _, nchunks, _ = dt.unfold(1, chunk_size, chunk_size).shape
    dt_out = torch.empty(nchunks, *dt.shape, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(nchunks, *A.shape, device=A.device, dtype=torch.float32)
    if has_heads:
        grid = lambda META: (
            batch,
            nchunks,
            triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
        )
    else:
        grid = lambda META: (batch, nchunks, 1)
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid](
            dt,
            A,
            dt_bias,
            dt_out,
            dA_cumsum,
            chunk_size,
            dt_limit[0],
            dt_limit[1],
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0),
            dt_out.stride(2),
            dt_out.stride(1),
            dt_out.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    if not has_heads:
        dt_out = dt_out.squeeze(2)
        dA_cumsum = dA_cumsum.squeeze(2)
    return dA_cumsum, dt_out

def _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=None, dt_softplus=False):
    has_heads = dt.dim() == 3
    if dt.dim() == 2:
        dt = dt.unsqueeze(2)
    elif dt.dim() > 3:
        raise ValueError(f"dt must be 2 or 3 dimenstional, got {dt.dim()}")
    if ddt_out.dim() == 3:
        ddt_out = ddt_out.unsqueeze(2)
    elif ddt_out.dim() > 4:
        raise ValueError(f"ddt_out must be 3 or 4 dimenstional, got {ddt_out.dim()}")
    if A.dim() == 1:
        A = A.unsqueeze(0)
    elif A.dim() > 2:
        raise ValueError(f"A must be 1 or 2 dimenstional, got {A.dim()}")
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    elif dt_bias is not None and dt_bias.dim() == 3:
        dt_bias = dt_bias.unsqueeze(1)
    elif dt_bias is not None and dt_bias.dim() > 4:
        raise ValueError(f"dt_bias must be 1, 2, or 4 dimenstional, got {dt_bias.dim()}")
    batch,
