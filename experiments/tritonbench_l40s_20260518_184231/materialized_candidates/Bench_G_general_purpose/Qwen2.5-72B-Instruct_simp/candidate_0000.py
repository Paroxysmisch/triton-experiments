import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    N_CTX, N_HEAD, N_EMBD,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (N_CTX // BLOCK_SIZE)
    pid = pid % (N_CTX // BLOCK_SIZE)
    q_offset = bid * stride_qb + pid * BLOCK_SIZE
    k_offset = bid * stride_kb
    v_offset = bid * stride_vb
    o_offset = bid * stride_ob + pid * BLOCK_SIZE

    Q_block = tl.load(Q + q_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qm + tl.arange(0, N_EMBD)[None, :] * stride_qh)
    K_block = tl.load(K + k_offset + tl.arange(0, N_CTX)[:, None] * stride_km + tl.arange(0, N_EMBD)[None, :] * stride_kh)
    V_block = tl.load(V + v_offset + tl.arange(0, N_CTX)[:, None] * stride_vm + tl.arange(0, N_EMBD)[None, :] * stride_vh)

    QK = tl.dot(Q_block, K_block, trans_b=True)
    QK = tl.softmax(QK, axis=1)
    Out_block = tl.dot(QK, V_block)

    tl.store(Out + o_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_om + tl.arange(0, N_EMBD)[None, :] * stride_oh, Out_block)

@triton.jit
def _bwd_intra_kernel(
    Q, K, V, Out, GradOut, GradQ, GradK, GradV,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    stride_gob, stride_goh, stride_gom,
    stride_gqb, stride_gqh, stride_gqm,
    stride_gkb, stride_gkh, stride_gkm,
    stride_gvb, stride_gvh, stride_gvm,
    N_CTX, N_HEAD, N_EMBD,
    BLOCK_SIZE: tl.constexpr,
    CBLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (N_CTX // BLOCK_SIZE)
    pid = pid % (N_CTX // BLOCK_SIZE)
    q_offset = bid * stride_qb + pid * BLOCK_SIZE
    k_offset = bid * stride_kb
    v_offset = bid * stride_vb
    o_offset = bid * stride_ob + pid * BLOCK_SIZE
    gob_offset = bid * stride_gob + pid * BLOCK_SIZE
    gq_offset = bid * stride_gqb + pid * BLOCK_SIZE
    gk_offset = bid * stride_gkb
    gv_offset = bid * stride_gvb

    Q_block = tl.load(Q + q_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qm + tl.arange(0, N_EMBD)[None, :] * stride_qh)
    K_block = tl.load(K + k_offset + tl.arange(0, N_CTX)[:, None] * stride_km + tl.arange(0, N_EMBD)[None, :] * stride_kh)
    V_block = tl.load(V + v_offset + tl.arange(0, N_CTX)[:, None] * stride_vm + tl.arange(0, N_EMBD)[None, :] * stride_vh)
    Out_block = tl.load(Out + o_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_om + tl.arange(0, N_EMBD)[None, :] * stride_oh)
    GradOut_block = tl.load(GradOut + gob_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_gom + tl.arange(0, N_EMBD)[None, :] * stride_goh)

    QK = tl.dot(Q_block, K_block, trans_b=True)
    QK = tl.softmax(QK, axis=1)
    GradQK = tl.dot(GradOut_block, V_block, trans_b=True)
    GradQ = tl.dot(GradQK, K_block)
    GradK = tl.dot(Q_block, GradQK, trans_a=True)
    GradV = tl.dot(QK, GradOut_block, trans_a=True)

    tl.store(GradQ + gq_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_gqm + tl.arange(0, N_EMBD)[None, :] * stride_gqh, GradQ)
    tl.store(GradK + gk_offset + tl.arange(0, N_CTX)[:, None] * stride_gkm + tl.arange(0, N_EMBD)[None, :] * stride_gkh, GradK)
    tl.store(GradV + gv_offset + tl.arange(0, N_CTX)[:, None] * stride_gvm + tl.arange(0, N_EMBD)[None, :] * stride_gvh, GradV)

@triton.jit
def _bwd_inter_kernel(
    GradK, GradV,
    stride_gkb, stride_gkh, stride_gkm,
    stride_gvb, stride_gvh, stride_gvm,
    N_CTX, N_HEAD, N_EMBD,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (N_CTX // BLOCK_SIZE)
    pid = pid % (N_CTX // BLOCK_SIZE)
    gk_offset = bid * stride_gkb + pid * BLOCK_SIZE
    gv_offset = bid * stride_gvb + pid * BLOCK_SIZE

    GradK_block = tl.load(GradK + gk_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_gkm + tl.arange(0, N_EMBD)[None, :] * stride_gkh)
    GradV_block = tl.load(GradV + gv_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_gvm + tl.arange(0, N_EMBD)[None, :] * stride_gvh)

    GradK_sum = tl.sum(GradK_block, axis=0)
    GradV_sum = tl.sum(GradV_block, axis=0)

    tl.store(GradK + gk_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_gkm + tl.arange(0, N_EMBD)[None, :] * stride_gkh, GradK_sum)
    tl.store(GradV + gv_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_gvm + tl.arange(0, N_EMBD)[None, :] * stride_gvh, GradV_sum)
