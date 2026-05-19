import triton
import triton.language as tl

@triton.jit
def _attn_fwd(q, k, v, o, q_scale, k_scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(q.shape[0], BLOCK_M)
    grid_n = tl.cdiv(q.shape[1], BLOCK_N)
    row = pid // grid_n
    col = pid % grid_n

    q_ptr = q + row * q.stride(0) + col * q.stride(1)
    k_ptr = k + row * k.stride(0) + col * k.stride(1)
    v_ptr = v + row * v.stride(0) + col * v.stride(1)
    o_ptr = o + row * o.stride(0) + col * o.stride(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(q.shape[2]):
        qk_ptr = q_ptr + i * q.stride(2)
        qk = tl.load(qk_ptr, mask=tl.arange(BLOCK_M) < q.shape[2], other=0.0)
        qk *= q_scale

        kv_ptr = k_ptr + i * k.stride(2)
        kv = tl.load(kv_ptr, mask=tl.arange(BLOCK_M) < q.shape[2], other=0.0)
        kv *= k_scale

        qk = tl.dot(qk, kv.T)
        qk = tl.exp(qk)
        acc += qk

    tl.store(o_ptr, acc, mask=tl.arange(BLOCK_M) < o.shape[0] and tl.arange(BLOCK_N) < o.shape[1])
