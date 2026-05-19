import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 128
N_CTX = 1024  # Context size

@triton.jit
def _attn_fwd_inner(q, k, v, o, l_i, acc, q_scale, k_scale, q_offset, k_offset, v_offset, o_offset, n_heads, head_dim, BLOCK_SIZE, N_CTX):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    q_idx = q_offset + block_start + tl.arange(0, BLOCK_SIZE)
    k_idx = k_offset + block_start + tl.arange(0, BLOCK_SIZE)
    v_idx = v_offset + block_start + tl.arange(0, BLOCK_SIZE)
    o_idx = o_offset + block_start + tl.arange(0, BLOCK_SIZE)

    q_block = tl.load(q + q_idx)
    k_block = tl.load(k + k_idx)
    v_block = tl.load(v + v_idx)

    q_block = q_block * q_scale
    k_block = k_block * k_scale

    for i in range(0, N_CTX, BLOCK_SIZE):
        k_i = k_block + i
        v_i = v_block + i
        qk = tl.dot(q_block, k_i, allow_tf32=True)
        qk = tl.softmax(qk)
        o_block = tl.dot(qk, v_i, allow_tf32=True)
        l_i += tl.sum(qk, axis=1)
        acc += o_block

    tl.store(o + o_idx, acc)

@triton.jit
def _attn_fwd(q, k, v, o, q_scale, k_scale, n_heads, head_dim, BLOCK_SIZE, N_CTX):
    pid = tl.program_id(0)
    head_idx = pid // (N_CTX // BLOCK_SIZE)
    block_idx = pid % (N_CTX // BLOCK_SIZE)

    q_offset = head_idx * head_dim * N_CTX + block_idx * BLOCK_SIZE
    k_offset = head_idx * head_dim * N_CTX + block_idx * BLOCK_SIZE
    v_offset = head_idx * head_dim * N_CTX + block_idx * BLOCK_SIZE
    o_offset = head_idx * head_dim * N_CTX + block_idx * BLOCK_SIZE

    l_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE, head_dim), dtype=tl.float32)

    _attn_fwd_inner(q, k, v, o, l_i, acc, q_scale, k_scale, q_offset, k_offset, v_offset, o_offset, n_heads, head_dim, BLOCK_SIZE, N_CTX)

def forward(q, k, v, q_scale, k_scale, n_heads, head_dim):
    BLOCK_SIZE = 128
    N_CTX = 1024
    o = triton.testing.empty_like(q)

    grid = (n_heads * (N_CTX // BLOCK_SIZE),)
    _attn_fwd[grid](q, k, v, o, q_scale, k_scale, n_heads, head_dim, BLOCK_SIZE, N_CTX)

    return o
