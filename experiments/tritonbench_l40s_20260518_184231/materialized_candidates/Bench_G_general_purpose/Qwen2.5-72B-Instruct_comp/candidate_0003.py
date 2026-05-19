import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    n_heads, n_ctx, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (n_heads * n_ctx)
    hid = (pid % (n_heads * n_ctx)) // n_ctx
    mid = (pid % (n_heads * n_ctx)) % n_ctx

    # Pointers to Q, K, V
    Q += bid * stride_qb + hid * stride_qh + mid * stride_qm
    K += bid * stride_kb + hid * stride_kh
    V += bid * stride_vb + hid * stride_vh
    Out += bid * stride_ob + hid * stride_oh + mid * stride_om

    # Load Q: [BLOCK_M, BLOCK_D]
    q = tl.load(Q + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_D)[None, :] * head_dim)

    # Initialize output buffer
    out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Compute attention scores
    for n in range(0, n_ctx, BLOCK_N):
        # Load K: [BLOCK_N, BLOCK_D]
        k = tl.load(K + (n + tl.arange(0, BLOCK_N))[:, None] * stride_km + tl.arange(0, BLOCK_D)[None, :] * head_dim)
        # Compute dot product: [BLOCK_M, BLOCK_N]
        logits = tl.dot(q, k.T) * (1.0 / tl.sqrt(head_dim))
        # Apply softmax
        logits = tl.softmax(logits, axis=1)
        # Load V: [BLOCK_N, BLOCK_D]
        v = tl.load(V + (n + tl.arange(0, BLOCK_N))[:, None] * stride_vm + tl.arange(0, BLOCK_D)[None, :] * head_dim)
        # Compute weighted sum: [BLOCK_M, BLOCK_D]
        out += tl.dot(logits, v)

    # Store output
    tl.store(Out + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_D)[None, :] * head_dim, out)

### Wrapper Function
