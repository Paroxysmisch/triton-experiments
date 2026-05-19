triton
@triton.jit
def _fwd_kernel_aligned(
    Q: float32[*, *, :],  # [batch_size, num_heads, seq_len, d_model]
    K: float32[*, *, :],  # [batch_size, num_heads, seq_len, d_model]
    V: float32[*, *, :],  # [batch_size, num_heads, seq_len, d_model]
    B0: float32[*, *, :],  # [batch_size, num_heads, seq_len, d_model]
    Out: float32[*, *, :],  # [batch_size, num_heads, seq_len, d_model]
    sm_scale: float32,
    BLOCK_M: int32,
    BLOCK_N: int32,
    BLOCK_DMODEL: int32,
    BIAS_LAST_SIZE: int32,
    num_warps: int32,
    num_stages: int32
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    grid_m = tl.cdiv(Q.shape[2], BLOCK_M)
    grid_n = tl.cdiv(Q.shape[2], BLOCK_N)
    m = pid % grid_m
    n = pid // grid_m
    head = bid
    q = tl.load(Q + m * BLOCK_M * BLOCK_DMODEL + head * BLOCK_DMODEL, mask=m < Q.shape[2] and head < Q.shape[1], other=0.0)
    k = tl.load(K + n * BLOCK_N * BLOCK_DMODEL + head * BLOCK_DMODEL, mask=n < K.shape[2] and head < K.shape[1], other=0.0)
    v = tl.load(V + n * BLOCK_N * BLOCK_DMODEL + head * BLOCK_DMODEL, mask=n < V.shape[2] and head < V.shape[1], other=0.0)
    b0 = tl.load(B0 + n * BLOCK_N * BLOCK_DMODEL + head * BLOCK_DMODEL, mask=n < B0.shape[2] and head < B0.shape[1], other=0.0)
    out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d in range(0, Q.shape[3], BLOCK_DMODEL):
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk *= sm_scale
        qk += b0
        qk = tl.math.exp2(qk)
        qk /= tl.sum(qk, axis=1, keepdims=True)
        out += tl.dot(qk, v)
    tl.store(Out + m * BLOCK_M * BLOCK_DMODEL + n * BLOCK_N * BLOCK_DMODEL + head * BLOCK_DMODEL, out, mask=m < Out.shape[2] and n < Out.shape[2] and head < Out.shape[1])
