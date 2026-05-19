: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        rows = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)

def rms_norm_fwd_fused(x, weight, eps):
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    weight = weight.contiguous()
    mean = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    assert x.is_contiguous()
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 4096")
    with torch.cuda.device(x.device.index):
        _rms_norm_fwd_fused[(M,)](
            x,
            out,
            weight,
            mean,
            rstd,
            x.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    out = out.contiguous()
    return out, mean, rstd

def rms_norm_bwd_dx_fused(
    dx, dy, dw, x, weight, mean, rstd, eps, group_size_m
):
    if isinstance(dx, (np.ndarray, np.generic)):
        dx = torch.from_numpy(dx)
    if isinstance(dy, (np.ndarray, np.generic)):
        dy = torch.from_numpy(dy)
    if isinstance(x, (np.ndarray, np.generic)):
        x = torch.from_numpy(x)
    if isinstance(weight, (np.ndarray, np.generic)):
        weight = torch.from_numpy(weight)
    if isinstance(mean, (np.ndarray, np.generic)):
        mean = torch.from_numpy(mean)
    if isinstance(rstd, (np.ndarray, np.generic)):
        rstd = torch.from_numpy(rstd)
    if isinstance(group_size_m, (np.ndarray, np.generic)):
        group_size_m = torch.from_numpy(group_size_m)
    if isinstance(dw, (np.ndarray, np.generic)):
        dw = torch.from_numpy(dw)
    M, N = x.shape
    BLOCK_SIZE_N = 128
    GROUP_SIZE_M = 64
    if N <= 128:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    if dx.stride(-1) != 1:
        dx = dx.contiguous()
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    lock = torch.zeros((group_size_m,), dtype=torch.int32, device=x.device)
    with torch.cuda.device(x.device.index):
        _rms_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            dw,
            x,
            weight,
            mean,
            rstd,
            lock,
            x.stride(0),
            N,
            eps,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    return dx

def rms_norm_bwd_dwdb(dw, final_dw, M, N):
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    if N <= 128:
        BLOCK_SIZE_N = triton.next_power_of_2(N)
    dw = dw.contiguous()
    final_dw = final_dw.contiguous()
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with torch.cuda.device(dw.device.index):
        _rms_norm_bwd_dwdb[grid](
            dw,
            final_dw,
            M,
            N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    return final_dw
