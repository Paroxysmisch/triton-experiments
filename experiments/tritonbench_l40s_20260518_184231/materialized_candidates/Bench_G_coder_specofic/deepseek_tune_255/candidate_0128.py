import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X,
    OUT,
    COS,
    SIN,
    seqlens_offsets,
    seqlens_offsets_conj,
    CU_SEQLENS,
    seqlens_in_samples,
    seqlens_in_samples_conj,
    seqlen_block_size,
    nheads,
    rotary_dim,
    interleaved,
    inplace,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Kernel logic
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    m_range_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_range_offs = tl.arange(0, rotary_dim // 2)
    head_range_offs = pid_head * tl.cdiv(rotary_dim, 2) + dim_range_offs

    offs_x = pid_batch * seqlen_block_size * nheads * rotary_dim + \
        m_range_offs[None, None, :] * nheads * rotary_dim + \
        head_range_offs[None, :, None]
    offs_y = pid_batch * seqlen_block_size * nheads * rotary_dim + \
        m_range_offs[None, None, :] * nheads * rotary_dim + \
        (head_range_offs + rotary_dim // 2)[None, :, None]

    if not seqlens_in_samples:
        mask_x = m_range_offs[:, None] < seqlen_block_size
        if CU_SEQLENS:
            mask_x = mask_x & (pid_m < seqlens_offsets[pid_batch + 1] -
                               seqlens_offsets[pid_batch])
        else:
            mask_x = mask_x & (pid_m < seqlen_block_size)

    if not seqlens_in_samples_conj:
        mask_y = m_range_offs[:, None] < seqlen_block_size
        if CU_SEQLENS:
            mask_y = mask_y & (pid_m < seqlens_offsets_conj[pid_batch + 1] -
                               seqlens_offsets_conj[pid_batch])
        else:
            mask_y = mask_y & (pid_m < seqlen_block_size)

    load_x = tl.load(X + offs_x, mask=mask_x, other=0.0)
    if CONJUGATE:
        load_x = tl.conj(load_x)

    if interleaved:
        rotary_dim = rotary_dim // 2
        load_cos = tl.load(
            COS + pid_batch * rotary_dim // 2 * 2 * 2 +
            m_range_offs[:, None] * 2 * 2 + head_range_offs[None, :, None] +
            rotary_dim // 2,
            mask=mask_x,
            other=0.0,
        )
        load_sin = tl.load(
            SIN + pid_batch * rotary_dim // 2 * 2 * 2 +
            m_range_offs[:, None] * 2 * 2 + head_range_offs[None, :, None],
            mask=mask_x,
            other=0.0,
        )
    else:
        load_cos = tl.load(
            COS + pid_batch * rotary_dim * 2 * 2 + m_range_offs[:, None] * 2 *
            2 + head_range_offs[None, :, None],
            mask=mask_x,
            other=0.0,
        )
        load_sin = tl.load(
            SIN + pid_batch * rotary_dim * 2 * 2 + m_range_offs[:, None] * 2 *
            2 + head_range_offs[None, :, None] + rotary_dim // 2,
            mask=mask_x,
            other=0.0,
        )

    if not inplace:
        if CONJUGATE:
            out_x = load_cos * tl.conj(load_x) - load_sin * load_x
        else:
            out_x = load_cos * load_x - load_sin * tl.conj(load_x)
        tl.store(OUT + offs_x, out_x, mask=mask_x)
        tl.store(OUT + offs_y, load_sin, mask=mask_y)
    else:
        out_x = load_cos * load_x - load_sin * tl.conj(load_x)
        tl.store(X + offs_x, out_x, mask=mask_x)
        tl.store(X + offs_y, load_sin, mask=mask_y)


def apply_rotary(x,
                 cos,
                 sin,
                 seqlens_offsets,
                 seqlens_offsets_conj=None,
                 cu_seqlens=None,
                 seqlens_in_samples=False,
                 seqlens_in_samples_conj=False,
                 interleaved=False,
                 inplace=False,
                 conjugate=False):
    if seqlens_offsets_conj is None:
        seqlens_offsets_conj = seqlens_offsets

    batch, seqlen, nheads, nheads_dim = x.shape
    assert nheads_dim % 2 == 0
    rotary_dim = nheads_dim // 2
    assert cos.shape == sin.shape
    assert cos.shape[0] == batch
    assert cos.shape[1] == rotary_dim
    assert cos.shape[2] == nheads * 2
    assert cos.shape[3] == 2

    out = x if inplace else torch.empty_like(x)

    if cu_seqlens is not None:
        assert seqlens_offsets.dim() == 1
        assert seqlens_offsets_conj.dim() == 1
        assert cu_seqlens.dim() == 1
        assert cu_seqlens[-1] == batch + 1
        assert seqlens_offsets.shape[0] == batch
        assert seqlens_offsets_conj.shape[0] == batch
    else:
        assert seqlens_offsets.shape[0] == batch + 1
        assert seqlens_offsets_conj.shape[0] == batch + 1

    grid = lambda META: (
        triton.cdiv(seqlen, META['BLOCK_M']),
        nheads,
        batch,
    )

    rotary_kernel[grid](
        x,
        out,
        cos,
        sin,
        seqlens_offsets,
        seqlens_offsets_conj,
        cu_seqlens is not None,
        seqlens_in_samples,
        seqlens_in_samples_conj,
        seqlen,
        nheads,
        rotary_dim,
        interleaved,
        inplace,
        CONJUGATE=conjugate,
    )
    return out
