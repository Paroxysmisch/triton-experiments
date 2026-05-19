import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    q_ptr, k_ptr, v_ptr, h_ptr, init_state_ptr, 
    q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1, h_stride0, h_stride1, 
    init_state_stride0, init_state_stride1, 
    N_CTX, N_HEAD, N_DIM, scale, decay, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_CTX

    q_offsets = pid * q_stride0 + tl.arange(0, N_DIM)
    k_offsets = pid * k_stride0 + tl.arange(0, N_DIM)
    v_offsets = pid * v_stride0 + tl.arange(0, N_DIM)
    h_offsets = pid * h_stride0 + tl.arange(0, N_DIM)

    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=mask, other=0.0)

    if pid == 0:
        init_state = tl.load(init_state_ptr + tl.arange(0, N_DIM), other=0.0)
    else:
        prev_h = tl.load(h_ptr + (pid - 1) * h_stride0 + tl.arange(0, N_DIM), other=0.0)
        init_state = prev_h * decay

    h = q * k * v * scale + init_state
    tl.store(h_ptr + h_offsets, h, mask=mask)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr, 
    q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1, h_stride0, h_stride1, o_stride0, o_stride1, 
    N_CTX, N_HEAD, N_DIM, scale, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_CTX

    q_offsets = pid * q_stride0 + tl.arange(0, N_DIM)
    k_offsets = pid * k_stride0 + tl.arange(0, N_DIM)
    v_offsets = pid * v_stride0 + tl.arange(0, N_DIM)
    h_offsets = pid * h_stride0 + tl.arange(0, N_DIM)
    o_offsets = pid * o_stride0 + tl.arange(0, N_DIM)

    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=mask, other=0.0)
    h = tl.load(h_ptr + h_offsets, mask=mask, other=0.0)

    o = q * k * v * scale + h
    tl.store(o_ptr + o_offsets, o, mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q_ptr, k_ptr, v_ptr, h_ptr, dh_ptr, 
    q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1, h_stride0, h_stride1, dh_stride0, dh_stride1, 
    N_CTX, N_HEAD, N_DIM, scale, decay, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_CTX

    q_offsets = pid * q_stride0 + tl.arange(0, N_DIM)
    k_offsets = pid * k_stride0 + tl.arange(0, N_DIM)
    v_offsets = pid * v_stride0 + tl.arange(0, N_DIM)
    h_offsets = pid * h_stride0 + tl.arange(0, N_DIM)
    dh_offsets = pid * dh_stride0 + tl.arange(0, N_DIM)

    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=mask, other=0.0)
    h = tl.load(h_ptr + h_offsets, mask=mask, other=0.0)

    dh = tl.load(dh_ptr + dh_offsets, mask=mask, other=0.0)
    dh = dh * scale

    if pid > 0:
        prev_h = tl.load(h_ptr + (pid - 1) * h_stride0 + tl.arange(0, N_DIM), other=0.0)
        dh += prev_h * decay

    tl.store(dh_ptr + dh_offsets, dh, mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q_ptr, k_ptr, v_ptr, h_ptr, dh_ptr, dq_ptr, dk_ptr, dv_ptr, 
    q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1, h_stride0, h_stride1, 
    dh_stride0, dh_stride1, dq_stride0, dq_stride1, dk_stride0, dk_stride1, dv_stride0, dv_stride1, 
    N_CTX, N_HEAD, N_DIM, scale, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_CTX

    q_offsets = pid * q_stride0 + tl.arange(0, N_DIM)
    k_offsets = pid * k_stride0 + tl.arange(0, N_DIM)
    v_offsets = pid * v_stride0 + tl.arange(0, N_DIM)
    h_offsets = pid * h_stride0 + tl.arange(0, N_DIM)
    dh_offsets = pid * dh_stride0 + tl.arange(0, N_DIM)
    dq_offsets = pid * dq_stride0 + tl.arange(0, N_DIM)
    dk_offsets = pid * dk_stride0 + tl.arange(0, N_DIM)
    dv_offsets = pid * dv_stride0 + tl.arange(0, N_DIM)

    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=mask, other=0.0)
    h = tl.load(h_ptr + h_offsets, mask=mask, other=0.0)
    dh = tl.load(dh_ptr + dh_offsets, mask=mask, other=0.0)

    dq = dh * k * v * scale
    dk = dh * q * v * scale
    dv = dh * q * k * scale

    tl.store(dq_ptr + dq_offsets, dq, mask=mask)
    tl.store(dk_ptr + dk_offsets, dk, mask=mask)
    tl.store(dv_ptr + dv_offsets, dv, mask=mask)
