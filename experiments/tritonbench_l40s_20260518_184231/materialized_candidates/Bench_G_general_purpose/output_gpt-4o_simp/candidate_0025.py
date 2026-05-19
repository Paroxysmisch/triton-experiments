import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, O, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, causal: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q = tl.load(Q + offs_m[:, None] * N + offs_n[None, :])
    k = tl.load(K + offs_m[:, None] * N + offs_n[None, :])
    v = tl.load(V + offs_m[:, None] * N + offs_n[None, :])

    # Compute the scaled dot-product attention
    scale = 1.0 / tl.sqrt(float(N))
    qk = tl.dot(q, k, trans_b=True) * scale

    # Apply causal masking if required
    if causal:
        mask = tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :]
        qk = tl.where(mask, qk, float('-inf'))

    # Softmax along the last dimension
    qk_max = tl.max(qk, axis=1)
    qk_exp = tl.exp(qk - qk_max[:, None])
    qk_sum = tl.sum(qk_exp, axis=1)
    attn = qk_exp / qk_sum[:, None]

    # Compute the output
    o = tl.dot(attn, v)
    tl.store(O + offs_m[:, None] * N + offs_n[None, :], o)

### Python Wrapper

The Python wrapper `flash_attn_triton` will handle the setup and execution of the Triton kernel.
