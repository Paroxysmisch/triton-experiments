import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    q, k, v, h, initial_state, final_state,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_hb, stride_hh, stride_hm,
    stride_isb, stride_ish, stride_ism,
    stride_fsb, stride_fsh, stride_fsm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads = q.shape[1]
    head_id = pid % num_heads
    block_id = pid // num_heads

    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q + (offs_m[:, None, None] * stride_qm + head_id * stride_qh + offs_d[None, None, :])
    k_ptrs = k + (offs_n[None, :, None] * stride_km + head_id * stride_kh + offs_d[None, None, :])
    v_ptrs = v + (offs_n[None, :, None] * stride_vm + head_id * stride_vh + offs_d[None, None, :])

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    h_ptrs = h + (offs_m[:, None, None] * stride_hm + head_id * stride_hh + offs_d[None, None, :])
    h = tl.zeros_like(h_ptrs)

    if initial_state is not None:
        initial_state_ptrs = initial_state + (head_id * stride_ish + offs_d[None, None, :])
        initial_state = tl.load(initial_state_ptrs)
        h += initial_state

    for n in range(BLOCK_N):
        h += tl.sum(q * k[n], axis=2)[:, :, None] * v[n]

    tl.store(h_ptrs, h)

    if final_state is not None:
        final_state_ptrs = final_state + (head_id * stride_fsh + offs_d[None, None, :])
        tl.store(final_state_ptrs, h[0, 0, :])

@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_hb, stride_hh, stride_hm,
    stride_ob, stride_oh, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads = q.shape[1]
    head_id = pid % num_heads
    block_id = pid // num_heads

    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q + (offs_m[:, None, None] * stride_qm + head_id * stride_qh + offs_d[None, None, :])
    h_ptrs = h + (offs_m[:, None, None] * stride_hm + head_id * stride_hh + offs_d[None, None, :])
    o_ptrs = o + (offs_m[:, None, None] * stride_om + head_id * stride_oh + offs_d[None, None, :])

    q = tl.load(q_ptrs)
    h = tl.load(h_ptrs)

    o = tl.zeros_like(o_ptrs)

    for n in range(BLOCK_N):
        o += tl.sum(q * h[n], axis=2)[:, :, None]

    tl.store(o_ptrs, o)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q, k, v, h, o, grad_o, grad_h,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_hb, stride_hh, stride_hm,
    stride_ob, stride_oh, stride_om,
    stride_gob, stride_goh, stride_gom,
    stride_ghb, stride_ghh, stride_ghm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads = q.shape[1]
    head_id = pid % num_heads
    block_id = pid // num_heads

    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q + (offs_m[:, None, None] * stride_qm + head_id * stride_qh + offs_d[None, None, :])
    h_ptrs = h + (offs_m[:, None, None] * stride_hm + head_id * stride_hh + offs_d[None, None, :])
    grad_o_ptrs = grad_o + (offs_m[:, None, None] * stride_gom + head_id * stride_goh + offs_d[None, None, :])
    grad_h_ptrs = grad_h + (offs_m[:, None, None] * stride_ghm + head_id * stride_ghh + offs_d[None, None, :])

    q = tl.load(q_ptrs)
    h = tl.load(h_ptrs)
    grad_o = tl.load(grad_o_ptrs)

    grad_h = tl.zeros_like(grad_h_ptrs)

    for n in range(BLOCK_N):
        grad_h += tl.sum(q * grad_o, axis=2)[:, :, None] * h[n]

    tl.store(grad_h_ptrs, grad_h)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q, k, v, h, o, grad_o, grad_q, grad_k, grad_v,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_hb, stride_hh, stride_hm,
    stride_ob, stride_oh, stride_om,
    stride_gob, stride_goh, stride_gom,
    stride_gqb, stride_gqh, stride_gqm,
    stride_gkb, stride_gkh, stride_gkm,
    stride_gvb, stride_gvh, stride_gvm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads = q.shape[1]
    head_id = pid % num_heads
    block_id = pid // num_heads

    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q + (offs_m[:, None, None] * stride_qm + head_id * stride_qh + offs_d[None, None, :])
    k_ptrs = k + (offs_n[None, :, None] * stride_km + head_id * stride_kh + offs_d[None, None, :])
    v_ptrs = v + (offs_n[None, :, None] * stride_vm + head_id * stride_vh + offs_d[None, None, :])
    h_ptrs = h + (offs_m[:, None, None] * stride_hm + head_id * stride_hh + offs_d[None, None, :])
    grad_o_ptrs = grad_o + (offs_m[:, None, None] * stride_gom + head_id * stride_goh + offs_d[None, None, :])
    grad_q_ptrs = grad_q + (offs_m[:, None, None] * stride_gqm + head_id * stride_gqh + offs_d[None, None, :])
    grad_k_ptrs = grad_k + (offs_n[None, :, None] * stride_gkm + head_id * stride_gkh + offs_d[None, None, :])
    grad_v_ptrs = grad_v + (offs_n[None, :, None] * stride_gvm + head_id * stride_gvh + offs_d[None, None, :])

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    h = tl.load(h_ptrs)
    grad_o = tl.load(grad_o_ptrs)

    grad_q = tl.zeros_like(grad_q_ptrs)
    grad_k = tl.zeros_like(grad_k_ptrs)
    grad_v = tl.zeros_like(grad_v_ptrs)

    for n in range(BLOCK_N):
        grad_q += tl.sum(grad_o * h[n], axis=2)[:, :, None] * k[n]
        grad_k += tl.sum(grad_o * q, axis=2)
