import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    Q, K, V, o, z, 
    stride_qb, stride_qh, stride_qm, 
    stride_kb, stride_kh, stride_km, 
    stride_vb, stride_vh, stride_vm, 
    stride_ob, stride_oh, stride_om, 
    stride_zb, stride_zh, 
    n_head, n_ctx, scale, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (n_head * n_ctx)
    hid = (pid % (n_head * n_ctx)) // n_ctx
    qid = (pid % (n_head * n_ctx)) % n_ctx

    Q_block_ptr = tl.make_block_ptr(
        base=Q, shape=(n_ctx, n_ctx), strides=(stride_qm, stride_qh),
        offsets=(qid * BLOCK_M, hid * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(n_ctx, n_ctx), strides=(stride_km, stride_kh),
        offsets=(0, hid * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(n_ctx, n_ctx), strides=(stride_vm, stride_vh),
        offsets=(0, hid * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=o, shape=(n_ctx, n_ctx), strides=(stride_om, stride_oh),
        offsets=(qid * BLOCK_M, hid * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    Z_block_ptr = tl.make_block_ptr(
        base=z, shape=(n_ctx, n_head), strides=(stride_zm, stride_zh),
        offsets=(qid, hid), block_shape=(1, 1), order=(1, 0)
    )

    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k in range(0, n_ctx, BLOCK_N):
        k_block_ptr = tl.advance(K_block_ptr, (k, 0))
        v_block_ptr = tl.advance(V_block_ptr, (k, 0))
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        acc += tl.dot(q, k, trans_b=True) * scale

    z = tl.max(acc, 1)
    z = tl.exp(z - z)
    z = tl.sum(z, 1)
    z = 1.0 / z
    tl.store(Z_block_ptr, z)

    o = tl.dot(acc, v)
    o = o * z
    tl.store(O_block_ptr, o)

@triton.jit
def parallel_rebased_bwd_kernel(
    Q, K, V, o, z, do, dq, dk, dv, 
    stride_qb, stride_qh, stride_qm, 
    stride_kb, stride_kh, stride_km, 
    stride_vb, stride_vh, stride_vm, 
    stride_ob, stride_oh, stride_om, 
    stride_zb, stride_zh, 
    stride_dob, stride_doh, stride_dom, 
    stride_dqb, stride_dqh, stride_dqm, 
    stride_dkb, stride_dkh, stride_dkm, 
    stride_dvb, stride_dvh, stride_dvm, 
    n_head, n_ctx, scale, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (n_head * n_ctx)
    hid = (pid % (n_head * n_ctx)) // n_ctx
    qid = (pid % (n_head * n_ctx)) % n_ctx

    Q_block_ptr = tl.make_block_ptr(
        base=Q, shape=(n_ctx, n_ctx), strides=(stride_qm, stride_qh),
        offsets=(qid * BLOCK_M, hid * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(n_ctx, n_ctx), strides=(stride_km, stride_kh),
        offsets=(0, hid * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(n_ctx, n_ctx), strides=(stride_vm, stride_vh),
        offsets=(0, hid * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=o, shape=(n_ctx, n_ctx), strides=(stride_om, stride_oh),
        offsets=(qid * BLOCK_M, hid * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    Z_block_ptr = tl.make_block_ptr(
        base=z, shape=(n_ctx, n_head), strides=(stride_zm, stride_zh),
        offsets=(qid, hid), block_shape=(1, 1), order=(1, 0)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=do, shape=(n_ctx, n_ctx), strides=(stride_dom, stride_doh),
        offsets=(qid * BLOCK_M, hid * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=dq, shape=(n_ctx, n_ctx), strides=(stride_dqm, stride_dqh),
        offsets=(qid * BLOCK_M, hid * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    DK_block_ptr = tl.make_block_ptr(
        base=dk, shape=(n_ctx, n_ctx), strides=(stride_dkm, stride_dkh),
        offsets=(0, hid * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=dv, shape=(n_ctx, n_ctx), strides=(stride_dvm, stride_dvh),
        offsets=(0, hid * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )

    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    o = tl.load(O_block_ptr)
    z = tl.load(Z_block_ptr)
    do = tl.load(DO_block_ptr)

    dz = tl.sum(do * o, 1)
    dz = dz * z * z
    dv = tl.dot(do, q, trans_a=True)
    dk = tl.dot(do, v, trans_b=True)
    dq = tl.dot(do, k)

    tl.store(DQ_block_ptr, dq)
    tl.store(DK_block_ptr, dk)
    tl.store(DV_block_ptr, dv)
