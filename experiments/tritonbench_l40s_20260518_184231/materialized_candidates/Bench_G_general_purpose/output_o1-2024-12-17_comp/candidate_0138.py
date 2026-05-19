import triton
import triton.language as tl


@triton.jit
def block_sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    layout_csr_row_indices_ptr, layout_csr_col_indices_ptr,
    B, M, N, D,
    num_heads, num_kv_heads,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr
):
    row_block_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    batch_id = bh_id // num_heads
    head_id = bh_id % num_heads
    kv_head_id = head_id % num_kv_heads

    q_offs_m = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, BLOCK_D)
    Q_block = tl.load(
        Q_ptr
        + batch_id * stride_qb
        + head_id * stride_qh
        + q_offs_m[:, None] * stride_qm
        + q_offs_d[None, :] * stride_qd,
        mask=(q_offs_m < M)[:, None] & (q_offs_d < D)[None, :],
        other=0.0
    )

    row_start = tl.load(layout_csr_row_indices_ptr + row_block_id)
    row_end = tl.load(layout_csr_row_indices_ptr + row_block_id + 1)
    nnz = row_end - row_start

    acc_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    for i in range(0, nnz):
        col_block_id = tl.load(layout_csr_col_indices_ptr + (row_start + i))
        n_offs = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        for dd in range(0, NUM_D_BLOCKS):
            k_offs_d = dd * BLOCK_D + tl.arange(0, BLOCK_D)
            K_block = tl.load(
                K_ptr
                + batch_id * stride_kb
                + kv_head_id * stride_kh
                + n_offs[:, None] * stride_kn
                + k_offs_d[None, :] * stride_kd,
                mask=(n_offs < N)[:, None] & (k_offs_d < D)[None, :],
                other=0.0
            )
            qk = tl.dot(Q_block, tl.trans(K_block)) * softmax_scale
            row_max = tl.max(qk, 1)
            acc_max = tl.maximum(acc_max, row_max)

    acc_exp_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    for i in range(0, nnz):
        col_block_id = tl.load(layout_csr_col_indices_ptr + (row_start + i))
        n_offs = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        for dd in range(0, NUM_D_BLOCKS):
            k_offs_d = dd * BLOCK_D + tl.arange(0, BLOCK_D)
            K_block = tl.load(
                K_ptr
                + batch_id * stride_kb
                + kv_head_id * stride_kh
                + n_offs[:, None] * stride_kn
                + k_offs_d[None, :] * stride_kd,
                mask=(n_offs < N)[:, None] & (k_offs_d < D)[None, :],
                other=0.0
            )
            qk = tl.dot(Q_block, tl.trans(K_block)) * softmax_scale
            row_max = acc_max
            qk = qk - row_max[:, None]
            exp_qk = tl.exp(qk)
            acc_exp_sum += tl.sum(exp_qk, 1)

    out_block = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for i in range(0, nnz):
        col_block_id = tl.load(layout_csr_col_indices_ptr + (row_start + i))
        n_offs = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        for dd in range(0, NUM_D_BLOCKS):
            d_offs = dd * BLOCK_D + tl.arange(0, BLOCK_D)
            K_block = tl.load(
                K_ptr
                + batch_id * stride_kb
                + kv_head_id * stride_kh
                + n_offs[:, None] * stride_kn
                + d_offs[None, :] * stride_kd,
                mask=(n_offs < N)[:, None] & (d_offs < D)[None, :],
                other=0.0
            )
            qk = (tl.dot(Q_block, tl.trans(K_block)) * softmax_scale) - acc_max[:, None]
            att = tl.exp(qk) / acc_exp_sum[:, None]
            V_block = tl.load(
                V_ptr
                + batch_id * stride_vb
                + kv_head_id * stride_vh
                + n_offs[:, None] * stride_vn
                + d_offs[None, :] * stride_vd,
                mask=(n_offs < N)[:, None] & (d_offs < D)[None, :],
                other=0.0
            )
            weighted_v = tl.dot(att, V_block)
            out_block += weighted_v

    mask_m = (q_offs_m < M)[:, None] & (q_offs_d < D)[None, :]
    tl.store(
        out_ptr
        + batch_id * stride_ob
        + head_id * stride_oh
        + q_offs_m[:, None] * stride_om
        + q_offs_d[None, :] * stride_od,
        out_block,
        mask=mask_m
    )


def block_sparse_attention(
    Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out,
    B, M, N, D,
    num_heads, num_kv_heads, softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    BLOCK_M, BLOCK_N, BLOCK_D, NUM_D_BLOCKS
):
    grid = ( (M + BLOCK_M - 1) // BLOCK_M, B * num_heads )
    block_sparse_attention_kernel[grid](
        Q, K, V, out,
        layout_csr_row_indices, layout_csr_col_indices,
        B, M, N, D,
        num_heads, num_kv_heads,
        softmax_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=NUM_D_BLOCKS
    )
