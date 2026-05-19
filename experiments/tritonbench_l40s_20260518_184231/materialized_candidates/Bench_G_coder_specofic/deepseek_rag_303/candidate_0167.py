import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q,
    K,
    V,
    K_scale,
    V,
    Out,
    q_scale,
    k_scale,
    sm_scale,
    N_CTX,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_kv_head_id = tl.program_id(0)
    cur_batch_id = tl.program_id(1)
    cur_q_head_id = cur_kv_head_id // N_CTX

    cur_kv_head_start_offset = cur_kv_head_id * BLOCK_N
    cur_kv_head_end_offset = cur_kv_head_start_offset + BLOCK_N

    cur_batch_start_offset = cur_batch_id * stride_qz

    block_mid = tl.arange(0, BLOCK_M)
    block_n = tl.arange(0, BLOCK_N)

    off_qm = cur_batch_start_offset + block_mid[:, None] * stride_qm
    off_qk = cur_kv_head_start_offset + block_n[None, :] * stride_qk

    off_km = block_n[:, None] * stride_kn + cur_kv_head_start_offset
    off_kk = cur_kv_head_start_offset + block_n[None, :] * stride_kk

    off_vk = cur_kv_head_start_offset + block_n[None, :] * stride_vk
    off_vn = block_n[:, None] * stride_vn + cur_kv_head_start_offset

    q = tl.load(
        Q + off_qm * stride_qm + off_qk, mask=(off_qm[:, 0] < cur_batch_end_loc) & (off_qk[0, :] < seq_len), other=0.0
    )

    k_ptrs = K + (off_km + off_kk)
    v_ptrs = V + (off_vk + off_vn)

    q *= q_scale
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    p = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    for start_n in range(0, BLOCK_N, 1):
        k = tl.load(k_ptrs, mask=(off_km[:, None] < cur_batch_end_loc) & (off_kk[None, :] < seq_len), other=0.0)
        v = tl.load(v_ptrs, mask=(off_vk[None, :] < cur_batch_end_loc) & (off_vn[:, None] < seq_len), other=0.0)
        if N_CTX == 1:
            k *= k_scale
        k *= k_scale
        pk = tl.dot(q.to(v.dtype), k)
        pk *= sm_scale
        p += pk

        qk += pk
        k_ptrs += stride_kn
        v_ptrs += stride_vn

    qk = qk.to(v.dtype)
    m_i = tl.max(qk, 1)
    p = p.to(qk.dtype)

    acc = tl.dot(qk.to(v.dtype), v)
    l_i = tl.cumsum(v, axis=1)
    l_i = tl.where(block_mid < (seq_len - 1), l_i, l_i - m_i[:, None] + tl.log(tl.sum(tl.exp(qk - m_i[:, None]), axis=1)))

    p = p.to(v.dtype)
    p = p / tl.exp(m_i[:, None] - l_i)
    qk = qk.to(v.dtype)

    acc_scale_denom = None
    for start_n in range(0, BLOCK_N, 1):
        k = tl.load(k_ptrs, mask=(off_km[:, None] < cur_batch_end_loc) & (off_kk[None, :] < seq_len), other=0.0)
        v = tl.load(v_ptrs, mask=(off_vk[None, :] < cur_batch_end_loc) & (off_vn[:, None] < seq_len), other=0.0)

        if acc_scale_denom is None:
            p = p / tl.exp(m_i[:, None] - l_i)
            acc_scale = (l_i - m_i[:, None])
            alpha = tl.exp(l_i - m_i[:, None])
            acc_scale_denom = tl.exp(m_i - l_i)
        else:
            acc_scale = acc_scale * acc_scale_denom + (l_i - m_i[:, None])
            alpha = alpha * tl.exp(m_i - l_i)
            acc_scale_denom = tl.exp(m_i - l_i) * acc_scale_denom

        l_i = l_i + tl.sum(v, axis=1)

        acc_scale = acc_scale.to(v.dtype)
        acc += tl.dot(p.to(v.dtype), v)

        p = p * alpha
        k_ptrs += stride_kn
        v_ptrs += stride_vn

    off_om = cur_batch_start_offset + block_mid[:, None] * stride_om
    off_on = cur_kv_head_start_offset + block_n[None, :] * stride_on
    out_ptrs = Out + (off_om + off_on)

    tl.store(out_ptrs, acc)

def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    V_scale,
    Out,
    qk_bias,
    q_scale,
    k_scale,
    sparse_counts,
    B_start_loc,
    B_seq_len,
    context_size,
):
    q_block, k_block = min(Q.shape[-1] // 128, 128), min(K.shape[-1] // 128, 128)
    assert q_block * q_block == q_block
    assert k_block * k_block == k_block

    batch, head, seq_len = B_seq_len.shape[0], K.shape[1], tl.max(B_seq_len)
    grid = (int(head // context_size), int(batch))
    num_warps = 2 if context_size == 1 else 4
