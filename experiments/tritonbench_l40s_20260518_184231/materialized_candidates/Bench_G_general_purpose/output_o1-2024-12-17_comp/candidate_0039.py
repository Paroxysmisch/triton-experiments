import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    batch_size, seq_len, head_dim, num_heads,
    log_scale, max_seqlen,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    bz = pid_batch_head // num_heads
    bh = pid_batch_head % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    Q_ptrs = Q_ptr + bz * stride_qz + bh * stride_qh + offs_m[:, None] * stride_qm
    K_ptrs = K_ptr + bz * stride_kz + bh * stride_kh + offs_n[None, :] * stride_kn
    V_ptrs = V_ptr + bz * stride_vz + bh * stride_vh
    Out_ptrs = Out_ptr + bz * stride_oz + bh * stride_oh + offs_m[:, None] * stride_om

    # Load partial Q
    q_vals = tl.load(Q_ptrs, mask=offs_m[:, None] < seq_len, other=0).to(tl.int8)
    q_vals_f32 = q_vals.to(tl.float32)

    # Compute attention scores
    scores = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for d_step in range(0, head_dim, BLOCK_DMODEL):
        k_vals = tl.load(K_ptrs + d_step, mask=(offs_n[None, :] < seq_len) & (d_step + offs_d[None, :] < head_dim), other=0).to(tl.int8)
        k_vals_f32 = k_vals.to(tl.float32)
        q_chunk = q_vals_f32[:, d_step : d_step + BLOCK_DMODEL]
        k_chunk = k_vals_f32[None, : , :]
        mat = tl.dot(q_chunk, tl.trans(k_chunk))
        scores += mat

    # Apply scaling
    scores = scores * log_scale

    # Causal mask
    mask = offs_m[:, None] >= offs_n[None, :]
    scores = tl.where(mask, scores, float('-inf'))

    # Softmax
    max_scores = tl.max(scores, 1)
    scores_exp = tl.exp(scores - max_scores[:, None])
    denom = tl.sum(scores_exp, 1)
    probs = scores_exp / denom[:, None]

    # Multiply by V
    out_vals = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
    for d_step in range(0, head_dim, BLOCK_DMODEL):
        v_vals = tl.load(V_ptrs + d_step + offs_n[None, :] * stride_vn, mask=(offs_n[None, :] < seq_len) & (d_step + offs_d[None, :] < head_dim), other=0).to(tl.int8)
        v_vals_f32 = v_vals.to(tl.float32)
        p_chunk = probs[:, :]
        v_chunk = v_vals_f32[:, :]
        out_vals += tl.dot(p_chunk, v_chunk)

    # Store
    tl.store(Out_ptrs, out_vals, mask=offs_m[:, None] < seq_len)


def context_attention_fwd_ppl_int8kv(Q, K, V, Out,
                                     batch_size, seq_len, head_dim, num_heads,
                                     stride_qz, stride_qh, stride_qm,
                                     stride_kz, stride_kh, stride_kn,
                                     stride_vz, stride_vh, stride_vn,
                                     stride_oz, stride_oh, stride_om,
                                     log_scale, max_seqlen,
                                     BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=32):
    grid = ( (seq_len + BLOCK_M - 1) // BLOCK_M, batch_size * num_heads )
    triton.run(
        _fwd_kernel_int8kv,
        grid=grid,
        num_warps=4,
        num_stages=2,
        Q_ptr=Q, K_ptr=K, V_ptr=V, Out_ptr=Out,
        stride_qz=stride_qz, stride_qh=stride_qh, stride_qm=stride_qm,
        stride_kz=stride_kz, stride_kh=stride_kh, stride_kn=stride_kn,
        stride_vz=stride_vz, stride_vh=stride_vh, stride_vn=stride_vn,
        stride_oz=stride_oz, stride_oh=stride_oh, stride_om=stride_om,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        num_heads=num_heads,
        log_scale=log_scale,
        max_seqlen=max_seqlen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
