import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qe,
    stride_kb, stride_kh, stride_ke,
    stride_vb, stride_vh, stride_ve,
    stride_ob, stride_oh, stride_oe,
    nheads, seq_len, embedding_dim, BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // nheads
    hid = pid % nheads

    q_start = bid * stride_qb + hid * stride_qh
    k_start = bid * stride_kb + hid * stride_kh
    v_start = bid * stride_vb + hid * stride_vh
    o_start = bid * stride_ob + hid * stride_oh

    for i in range(0, seq_len, BLOCK):
        q_block = tl.load(Q + q_start + i * stride_qe, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)
        k_block = tl.load(K + k_start + i * stride_ke, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)
        v_block = tl.load(V + v_start + i * stride_ve, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)

        attn = tl.dot(q_block, k_block, trans_b=True) * (1.0 / tl.sqrt(BLOCK))
        attn = tl.softmax(attn, axis=1)
        out_block = tl.dot(attn, v_block)

        tl.store(Out + o_start + i * stride_oe, out_block, mask=i + tl.arange(0, BLOCK) < seq_len)

@triton.jit
def _bwd_intra_kernel(
    Q, K, V, DO, DQ, DK, DV,
    stride_qb, stride_qh, stride_qe,
    stride_kb, stride_kh, stride_ke,
    stride_vb, stride_vh, stride_ve,
    stride_dob, stride_doh, stride_doe,
    stride_dqb, stride_dqh, stride_dqe,
    stride_dkb, stride_dkh, stride_dke,
    stride_dvb, stride_dvh, stride_dve,
    nheads, seq_len, embedding_dim, BLOCK: tl.constexpr, CBLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // nheads
    hid = pid % nheads

    q_start = bid * stride_qb + hid * stride_qh
    k_start = bid * stride_kb + hid * stride_kh
    v_start = bid * stride_vb + hid * stride_vh
    do_start = bid * stride_dob + hid * stride_doh
    dq_start = bid * stride_dqb + hid * stride_dqh
    dk_start = bid * stride_dkb + hid * stride_dkh
    dv_start = bid * stride_dvb + hid * stride_dvh

    for i in range(0, seq_len, BLOCK):
        q_block = tl.load(Q + q_start + i * stride_qe, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)
        k_block = tl.load(K + k_start + i * stride_ke, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)
        v_block = tl.load(V + v_start + i * stride_ve, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)
        do_block = tl.load(DO + do_start + i * stride_doe, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)

        attn = tl.dot(q_block, k_block, trans_b=True) * (1.0 / tl.sqrt(BLOCK))
        attn = tl.softmax(attn, axis=1)

        dv_block = tl.dot(attn, do_block, trans_a=True)
        do_block = tl.dot(attn, do_block)
        dq_block = tl.dot(do_block, k_block, trans_b=True) * (1.0 / tl.sqrt(BLOCK))
        dk_block = tl.dot(q_block, do_block, trans_a=True) * (1.0 / tl.sqrt(BLOCK))

        tl.store(DQ + dq_start + i * stride_dqe, dq_block, mask=i + tl.arange(0, BLOCK) < seq_len)
        tl.store(DK + dk_start + i * stride_dke, dk_block, mask=i + tl.arange(0, BLOCK) < seq_len)
        tl.store(DV + dv_start + i * stride_dve, dv_block, mask=i + tl.arange(0, BLOCK) < seq_len)

@triton.jit
def _bwd_inter_kernel(
    Q, K, V, DO, DQ, DK, DV,
    stride_qb, stride_qh, stride_qe,
    stride_kb, stride_kh, stride_ke,
    stride_vb, stride_vh, stride_ve,
    stride_dob, stride_doh, stride_doe,
    stride_dqb, stride_dqh, stride_dqe,
    stride_dkb, stride_dkh, stride_dke,
    stride_dvb, stride_dvh, stride_dve,
    nheads, seq_len, embedding_dim, BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // nheads
    hid = pid % nheads

    q_start = bid * stride_qb + hid * stride_qh
    k_start = bid * stride_kb + hid * stride_kh
    v_start = bid * stride_vb + hid * stride_vh
    do_start = bid * stride_dob + hid * stride_doh
    dq_start = bid * stride_dqb + hid * stride_dqh
    dk_start = bid * stride_dkb + hid * stride_dkh
    dv_start = bid * stride_dvb + hid * stride_dvh

    for i in range(0, seq_len, BLOCK):
        for j in range(0, seq_len, BLOCK):
            if i != j:
                q_block = tl.load(Q + q_start + i * stride_qe, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)
                k_block = tl.load(K + k_start + j * stride_ke, mask=j + tl.arange(0, BLOCK) < seq_len, other=0.0)
                v_block = tl.load(V + v_start + j * stride_ve, mask=j + tl.arange(0, BLOCK) < seq_len, other=0.0)
                do_block = tl.load(DO + do_start + i * stride_doe, mask=i + tl.arange(0, BLOCK) < seq_len, other=0.0)

                attn = tl.dot(q_block, k_block, trans_b=True) * (1.0 / tl.sqrt(BLOCK))
                attn = tl.softmax(attn, axis=1)

                dv_block = tl.dot(attn, do_block, trans_a=True)
                do_block = tl.dot(attn, do_block)
                dq_block = tl.dot(do_block, k_block, trans_b=True) * (1.0 / tl.sqrt(BLOCK))
                dk_block = tl.dot(q_block, do_block, trans_a=True) * (1.0 / tl.sqrt(BLOCK))

                tl.atomic_add(DQ + dq_start + i * stride_dqe, dq_block, mask=i + tl.arange(0, BLOCK) < seq_len)
                tl.atomic_add(DK + dk_start + j * stride_dke, dk_block, mask=j + tl.arange(0, BLOCK) < seq_len)
                tl.atomic_add(DV + dv_start + j * stride_dve, dv_block, mask=j + tl.arange(0, BLOCK) < seq_len)
