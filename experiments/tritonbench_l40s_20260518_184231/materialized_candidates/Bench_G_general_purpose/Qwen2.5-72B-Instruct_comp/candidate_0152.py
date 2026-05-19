import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr, b_h_ptr, initial_state_ptr, final_state_ptr,
    NT, NK, NV, NB, USE_INITIAL_STATE, STORE_FINAL_STATE,
    decay_factor, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    b_h = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)

    for t in range(NT):
        k_block = tl.load(k_ptr + t * NK + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        v_block = tl.load(v_ptr + t * NV + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)

        d_b = decay_factor ** t
        d_i = decay_factor ** (t + 1)

        b_h = b_h * d_b + tl.dot(k_block, v_block) * d_i

        if STORE_FINAL_STATE and t == NT - 1:
            tl.store(final_state_ptr + block_start * NB, b_h, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)

    tl.store(b_h_ptr + block_start * NB, b_h, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr, b_o_ptr, b_s_ptr,
    NT, NQ, NK, NV, NB, decay_factor, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    b_o = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)
    b_s = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)

    for t in range(NT):
        q_block = tl.load(q_ptr + t * NQ + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        k_block = tl.load(k_ptr + t * NK + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        v_block = tl.load(v_ptr + t * NV + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        h_block = tl.load(h_ptr + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)

        d_i = decay_factor ** (t + 1)

        b_o += tl.dot(q_block, k_block) * h_block * d_i
        b_s += tl.dot(q_block, k_block) * d_i

    o_block = b_o / (b_s + 1e-6)
    tl.store(o_ptr + block_start * NB, o_block, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    grad_o_ptr, q_ptr, k_ptr, v_ptr, h_ptr, grad_h_ptr,
    NT, NQ, NK, NV, NB, decay_factor, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    grad_h = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)

    for t in range(NT - 1, -1, -1):
        grad_o_block = tl.load(grad_o_ptr + t * NQ + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        q_block = tl.load(q_ptr + t * NQ + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        k_block = tl.load(k_ptr + t * NK + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        v_block = tl.load(v_ptr + t * NV + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        h_block = tl.load(h_ptr + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)

        d_i = decay_factor ** (t + 1)

        grad_h += tl.dot(tl.dot(grad_o_block, q_block), k_block) * h_block * d_i

    tl.store(grad_h_ptr + block_start * NB, grad_h, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    grad_o_ptr, q_ptr, k_ptr, v_ptr, h_ptr, grad_q_ptr, grad_k_ptr, grad_v_ptr,
    NT, NQ, NK, NV, NB, decay_factor, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    grad_q = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)
    grad_k = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)
    grad_v = tl.zeros((BLOCK_SIZE, NB), dtype=tl.float32)

    for t in range(NT):
        grad_o_block = tl.load(grad_o_ptr + t * NQ + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        q_block = tl.load(q_ptr + t * NQ + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        k_block = tl.load(k_ptr + t * NK + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        v_block = tl.load(v_ptr + t * NV + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)
        h_block = tl.load(h_ptr + block_start * NB, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT, other=0.0)

        d_i = decay_factor ** (t + 1)

        grad_q += tl.dot(grad_o_block, h_block) * k_block * d_i
        grad_k += tl.dot(grad_o_block, h_block) * q_block * d_i
        grad_v += tl.dot(grad_o_block, q_block) * k_block * d_i

    tl.store(grad_q_ptr + block_start * NB, grad_q, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)
    tl.store(grad_k_ptr + block_start * NB, grad_k, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)
    tl.store(grad_v_ptr + block_start * NB, grad_v, mask=block_start + tl.arange(0, BLOCK_SIZE) < NT)
