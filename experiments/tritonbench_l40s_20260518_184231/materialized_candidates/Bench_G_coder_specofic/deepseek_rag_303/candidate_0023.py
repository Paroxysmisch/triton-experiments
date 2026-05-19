import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(Q,
                            K,
                            COS,
                            SIN,
                            Q_EMB,
                            K_EMB,
                            stride_qs: tl.constexpr,
                            stride_qh: tl.constexpr,
                            stride_qd: tl.constexpr,
                            stride_ks: tl.constexpr,
                            stride_kh: tl.constexpr,
                            stride_kd: tl.constexpr,
                            stride_qes: tl.constexpr,
                            stride_qeh: tl.constexpr,
                            stride_qed: tl.constexpr,
                            stride_kes: tl.constexpr,
                            stride_keh: tl.constexpr,
                            stride_ked: tl.constexpr,
                            q_head_num: tl.constexpr,
                            K_N,
                            HEAD_DIM: tl.constexpr,
                            seq_len: tl.constexpr):
    qid = tl.program_id(axis=0)
    head_nid = tl.program_id(axis=1)

    head_off = head_nid * HEAD_DIM
    qv_off = (qid * stride_qs + head_off * stride_qh)
    block_base = qv_off // HEAD_DIM
    kv_block_id = block_base // K_N
    q_relative_base = block_base % K_N

    q_cos_addr = q_relative_base * stride_qd + head_off + 0
    q_sin_addr = q_relative_base * stride_qd + head_off + HEAD_DIM
    q_orig_addr = qv_off + head_off * stride_qh

    qe_cos_addr = q_relative_base * stride_qed + head_off + 0
    qe_sin_addr = q_relative_base * stride_qed + head_off + HEAD_DIM
    qe_orig_addr = qv_off + head_off * stride_qeh

    if q_head_num == head_nid + 1:
        HEAD_DIM_half = HEAD_DIM // 2
        q_l = tl.load(Q + q_orig_addr)
        q_h = tl.load(Q + q_orig_addr + HEAD_DIM_half * stride_qh)
        qe_l = tl.load(Q_EMB + qe_orig_addr)
        qe_h = tl.load(Q_EMB + qe_orig_addr + HEAD_DIM_half * stride_qeh)
    else:
        seq_block_id = tl.arange(0, 32)
        d_l_cos = seq_block_id * HEAD_DIM + head_off + 0
        d_l_sin = seq_block_id * HEAD_DIM + head_off + 0 + HEAD_DIM // 2
        d_h_cos = seq_block_id * HEAD_DIM + head_off + HEAD_DIM // 2
        d_h_sin = seq_block_id * HEAD_DIM + head_off + HEAD_DIM // 2 + HEAD_DIM

        mask = seq_block_id < seq_len

        q_l = tl.load(Q + q_orig_addr + d_l_cos * stride_qh, mask)
        q_h = tl.load(Q + q_orig_addr + d_h_cos * stride_qh, mask)
        q_cos_l = tl.load(COS + q_cos_addr + d_l_cos, mask)
        q_cos_h = tl.load(COS + q_cos_addr + d_h_cos, mask)
        q_sin_l = tl.load(SIN + q_sin_addr + d_l_sin, mask)
        q_sin_h = tl.load(SIN + q_sin_addr + d_h_sin, mask)

        qe_l = tl.load(Q_EMB + qe_orig_addr + d_l_cos * stride_qeh, mask)
        qe_h = tl.load(Q_EMB + qe_orig_addr + d_h_cos * stride_qeh, mask)
        qe_cos_l = tl.load(COS + qe_cos_addr + d_l_cos, mask)
        qe_cos_h = tl.load(COS + qe_cos_addr + d_h_cos, mask)
        qe_sin_l = tl.load(SIN + qe_sin_addr + d_l_sin, mask)
        qe_sin_h = tl.load(SIN + qe_sin_addr + d_h_sin, mask)

    q_l_new = q_l * q_cos_l - q_h * q_sin_l
    q_h_new = q_h * q_cos_h + q_l * q_sin_h
    qe_l_new = qe_l * qe_cos_l - qe_h * qe_sin_l
    qe_h_new = qe_h * qe_cos_h + qe_l * qe_sin_h

    if q_head_num == head_nid + 1:
        tl.store(Q + q_orig_addr, q_l_new)
        tl.store(Q + q_orig_addr + HEAD_DIM_half * stride_qh, q_h_new)
        tl.store(Q_EMB + qe_orig_addr, qe_l_new)
        tl.store(Q_EMB + qe_orig_addr + HEAD_DIM_half * stride_qeh, qe_h_new)
    else:
        tl.store(Q + q_orig_addr + d_l_cos * stride_qh,
                 q_l_new,
                 mask=seq_block_id < seq_len)
        tl.store(Q + q_orig_addr + d_h_cos * stride_qh,
                 q_h_new,
                 mask=seq_block_id < seq_len)
        tl.store(Q_EMB + qe_orig_addr + d_l_cos * stride_qeh,
                 qe_l_new,
                 mask=seq_block_id < seq_len)
        tl.store(Q_EMB + qe_orig_addr + d_h_cos * stride_qeh,
                 qe_h_new,
                 mask=seq_block_id < seq_len)


@triton.jit
def fused_rotary_embedding_kernel_v2(K,
                                     K_EMB,
                                     Q_COS_SIN,
                                     STRIDE_BLOCK,
                                     KV_LENGTHS,
                                     kv_start_blockid,
                                     kv_length,
                                     BLOCK_TABLES,
                                     SEQ_LENGTH,
                                     stride_ks: tl.constexpr,
                                     stride_kh: tl.constexpr,
                                     stride_kd: tl.constexpr,
                                     stride_kes: tl.constexpr,
                                     stride_keh: tl.constexpr,
                                     stride_ked: tl.constexpr,
                                     q_head_num: tl.constexpr,
                                     k_head_num: tl.constexpr,
                                     K_N,
                                     stride_bs: tl.constexpr,
                                     HEAD_DIM: tl.constexpr,
                                     seq_len: tl.constexpr):
    qid = tl.program_id(axis=0)
    head_nid = tl.program_id(axis=1)

    batch_id = tl.load(K + stride_ks
