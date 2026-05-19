in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, res_ptr,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_chunk, stride_db_head, stride_db_csize_m, stride_db_csize_n,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    pid_ch = tl.program_id(axis=3)
    pid_h = pid_ch - pid_c * ngroups
    pid_k = tl.program_id(axis=0)

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head + pid_k * stride_dout_csize_m
    db_ptr += pid_b * stride_db_batch + pid_c * stride_db_chunk + pid_h * stride_db_head + pid_k * stride_db_csize_m

    offs_m = tl.arange(0, BLOCK_SIZE_M) + pid_k * BLOCK_SIZE_M
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_CS + tl.arange(0, BLOCK_SIZE_CS)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_m + offs_n[None, :] * stride_dout_csize_n)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_csize_m + offs_n[None, :] * stride_db_csize_n)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dout_chunk_size = min(chunk_size, seqlen - pid_c * chunk_size)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_CS)):
        if HAS_RESIDUAL:
            residual = tl.load(res_ptr + pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head + (offs_k[:, None] + k * BLOCK_SIZE_CS) * stride_res_k, mask=(offs_k[:, None] + k * BLOCK_SIZE_CS) < K, other=0.0)
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_CS) & (offs_m[:, None] < chunk_size_limit), other=0.0).to(dot_dtype)
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dout_chunk_size), other=0.0).to(dot_dtype)
        if HAS_RESIDUAL:
            dout += residual[:, None]
        acc += tl.dot(dout, a.trans())
        a_ptrs += BLOCK_SIZE_CS * stride_ak
        dout_ptrs += BLOCK_SIZE_N * stride_dout_csize_n

    tl.store(db_ptrs, acc.to(db_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False):
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    nchunks = math.ceil(seqlen / chunk_size)
    out_dtype = a.dtype
    a = a.contiguous()
    b = b.contiguous()
    if not has_groups:
        out = torch.empty(batch, nchunks, k, k, device=a.device, dtype=out_dtype)
    else:
        out = torch.empty(batch, nchunks, ngroups, k, k, device=a.device, dtype=out_dtype)
    if seq_idx is not None:
        seq_idx = seq_idx.contiguous()
    dot_dtype = out_dtype
    if out_dtype in [torch.float16, torch.bfloat16]:
        dot_dtype = torch.float32
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']), nchunks, META['ngroups'] if has_groups else 1)
    with torch.cuda.device(a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a, b, out, seq_idx,
            seqlen, chunk_size, k, ngroups if has_groups else 1,
            a.stride(0), a.stride(1), a.stride(2 if has_groups else 1), a.stride(3 if has_groups else 2),
            b.stride(0), b.stride(1), b.stride(2 if has_groups else 1), b.stride(3 if has_groups else 2),
            out.stride(0), out.stride(1), out
