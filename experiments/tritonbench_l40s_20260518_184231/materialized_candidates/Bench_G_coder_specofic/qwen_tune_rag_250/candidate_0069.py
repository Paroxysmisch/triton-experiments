=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value = att_value.to(m_i.dtype)
        att_value = tl.where(m_mask & n_mask, att_value, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(att_value, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(att_value - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        p = p.to(dtype)
        acc += p[:, :, None] * v
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o = acc.to(dtype)
    tl.store(o_ptrs, o, mask=m_mask[:, None])
    return

@torch.no_grad()
def _triton_mixed_sparse_attention(
    q, k, v, seqlens, block_count, block_offset, column_count, column_index, sm_scale, sparse_block, sparse_n
):
    BLOCK = 128
    NUM_WARPS = 4
    Lk = k.shape[-1]
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    dtype = triton.dtype_of(q)
    num_warps = NUM_WARPS
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        sparse_block, sparse_n, sparse_n,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
