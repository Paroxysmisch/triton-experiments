import math
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_q_bs, stride_q_h, stride_q_m, stride_q_d,
    stride_k_bs, stride_k_h, stride_k_m, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_m, stride_v_d,
    stride_o_bs, stride_o_h, stride_o_m, stride_o_d,
    seqlen_q, seqlen_k, head_dim, sm_scale,
    kv_head_group_num,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    head_idx = pid_h
    batch_idx = pid_b

    Q_block_ptr = Q + batch_idx * stride_q_bs \
                    + head_idx * stride_q_h \
                    + offs_m[:, None] * stride_q_m \
                    + tl.arange(0, head_dim)[None, :] * stride_q_d

    # Load queries
    Q_block = tl.load(Q_block_ptr, mask=(offs_m[:, None] < seqlen_q), other=0.0)

    # Partial accumulators for attention scores
    attn_score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    num_k_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    # Dot product with K in chunks
    for nk in range(num_k_blocks):
        offs_kn = nk * BLOCK_N + offs_n
        K_block_ptr = K + batch_idx * stride_k_bs \
                        + (head_idx // kv_head_group_num) * stride_k_h \
                        + offs_kn[None, :] * stride_k_m \
                        + tl.arange(0, head_dim)[:, None] * stride_k_d
        mask_k = offs_kn[None, :] < seqlen_k
        K_block = tl.load(K_block_ptr, mask=mask_k, other=0.0)
        # Dot: [BLOCK_M, head_dim] x [head_dim, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        attn_score += tl.sum(Q_block @ K_block, axis=1)

    # Apply scaling
    attn_score *= sm_scale

    # Causal or sequence mask: positions beyond seqlen_k or offs_n > offs_m
    # for standard masking. Adjust logic for real usage.
    col_idx = tl.arange(0, BLOCK_N)[None, :] + tl.zeros([BLOCK_M, 1], tl.int32) \
              + (pid_m * BLOCK_M)[:, None] * 0
    row_idx = offs_m[:, None]
    mask = (col_idx > row_idx) | (row_idx >= seqlen_q) | (col_idx >= seqlen_k)
    attn_score = tl.where(mask, float('-inf'), attn_score)

    # Compute softmax
    max_score = tl.max(attn_score, 1)
    attn_score = attn_score - max_score[:, None]
    exp_score = tl.exp(attn_score)
    denom = tl.sum(exp_score, 1)
    exp_score = exp_score / denom[:, None]

    # Multiply by V
    out_accum = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    for nk in range(num_k_blocks):
        offs_kn = nk * BLOCK_N + offs_n
        K_block_ptr = K + batch_idx * stride_k_bs \
                        + (head_idx // kv_head_group_num) * stride_k_h \
                        + offs_kn[None, :] * stride_k_m \
                        + tl.arange(0, head_dim)[:, None] * stride_k_d
        # load mask for correctness, not used here in this snippet
        mask_k = offs_kn[None, :] < seqlen_k

        # load V
        V_block_ptr = V + batch_idx * stride_v_bs \
                        + (head_idx // kv_head_group_num) * stride_v_h \
                        + offs_kn[None, :] * stride_v_m \
                        + tl.arange(0, head_dim)[:, None] * stride_v_d
        V_block = tl.load(V_block_ptr, mask=mask_k, other=0.0)

        # recompute partial softmax
        attn_part = exp_score[:, nk * BLOCK_N + offs_n]
        out_accum += attn_part @ V_block

    # Store to output
    out_ptr = Out + batch_idx * stride_o_bs \
                    + head_idx * stride_o_h \
                    + offs_m[:, None] * stride_o_m \
                    + tl.arange(0, head_dim)[None, :] * stride_o_d
    tl.store(out_ptr, out_accum, mask=(offs_m[:, None] < seqlen_q))

def context_attention_fwd(
    Q, K, V, Out,
    batch_dim, head_dim_q, seq_len_q, head_dim_kv, seq_len_kv,
    stride_q_bs, stride_q_h, stride_q_m, stride_q_d,
    stride_k_bs, stride_k_h, stride_k_m, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_m, stride_v_d,
    stride_o_bs, stride_o_h, stride_o_m, stride_o_d,
    kv_head_group_num=1
):
    device_props = triton.devices[Q.device.index]
    # Example Tesla detection: adjust BLOCK_M
    if 'Tesla' in device_props.name:
        BLOCK_M = 64
    else:
        BLOCK_M = 128

    BLOCK_N = 64
    sm_scale = 1.0 / math.sqrt(float(head_dim_q))  # typical
    # small shift to stay in range for exponent
    sm_scale = sm_scale * 1.4426950408889634

    grid = lambda META: (
        ( (seq_len_q + BLOCK_M - 1) // BLOCK_M ),  # M dimension
        batch_dim * head_dim_q,                   # head dimension
        1                                          # batch dimension
    )

    _fwd_kernel[grid](
        Q, K, V, Out,
        stride_q_bs, stride_q_h, stride_q_m, stride_q_d,
        stride_k_bs, stride_k_h, stride_k_m, stride_k_d,
        stride_v_bs, stride_v_h, stride_v_m, stride_v_d,
        stride_o_bs, stride_o_h, stride_o_m, stride_o_d,
        seq_len_q, seq_len_kv, head_dim_q,
        sm_scale, kv_head_group_num,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
