import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=1),
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr,
    final_state_ptr,      # optional, can be None if STORE_FINAL_STATE=False
    initial_state_ptr,    # optional, can be None if USE_INITIAL_STATE=False
    B, T, D,
    stride_k_b, stride_k_t, stride_k_d,
    stride_v_b, stride_v_t, stride_v_d,
    stride_h_b, stride_h_t, stride_h_d,
    stride_fs_b, stride_fs_d,
    stride_is_b, stride_is_d,
    BLOCK_D: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    # program_id uniquely identifies [batch_idx, d_block]
    batch_idx = tl.program_id(0)
    d_block_idx = tl.program_id(1)
    block_offset_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = block_offset_d < D

    # Pointers
    k_offset = batch_idx * stride_k_b
    v_offset = batch_idx * stride_v_b
    h_offset = batch_idx * stride_h_b
    if USE_INITIAL_STATE:
        is_offset = batch_idx * stride_is_b
    if STORE_FINAL_STATE:
        fs_offset = batch_idx * stride_fs_b

    # Initialize hidden buffer
    b_h = tl.zeros([BLOCK_D], dtype=tl.float32)
    if USE_INITIAL_STATE:
        init_ptr = initial_state_ptr + is_offset + block_offset_d * stride_is_d
        b_init = tl.load(init_ptr, mask=mask_d, other=0.0)
        b_h += b_init

    # Compute over time dimension
    for t in range(T):
        # Load k, v
        k_ptr_t = k_ptr + k_offset + t * stride_k_t + block_offset_d * stride_k_d
        v_ptr_t = v_ptr + v_offset + t * stride_v_t + block_offset_d * stride_v_d
        b_k = tl.load(k_ptr_t, mask=mask_d, other=0.0)
        b_v = tl.load(v_ptr_t, mask=mask_d, other=0.0)

        # Example decay factors (user-defined decay function could replace these)
        d_b = 0.9
        d_i = 0.1

        # Update b_h
        b_h = b_h * d_b + b_k * b_v * d_i

        # Store intermediate h
        h_ptr_t = h_ptr + h_offset + t * stride_h_t + block_offset_d * stride_h_d
        tl.store(h_ptr_t, b_h, mask=mask_d)

    if STORE_FINAL_STATE:
        fs_ptr = final_state_ptr + fs_offset + block_offset_d * stride_fs_d
        tl.store(fs_ptr, b_h, mask=mask_d)


@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr,
    o_ptr, s_ptr,   # s_ptr is an auxiliary buffer for scaling or partial sums
    B, T, D,
    stride_q_b, stride_q_t, stride_q_d,
    stride_k_b, stride_k_t, stride_k_d,
    stride_v_b, stride_v_t, stride_v_d,
    stride_h_b, stride_h_t, stride_h_d,
    stride_o_b, stride_o_t, stride_o_d,
    stride_s_b, stride_s_t, stride_s_d,
    BLOCK_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    d_block_idx = tl.program_id(1)
    block_offset_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = block_offset_d < D

    q_offset = batch_idx * stride_q_b
    k_offset = batch_idx * stride_k_b
    v_offset = batch_idx * stride_v_b
    h_offset = batch_idx * stride_h_b
    o_offset = batch_idx * stride_o_b
    s_offset = batch_idx * stride_s_b

    for t in range(T):
        q_ptr_t = q_ptr + q_offset + t * stride_q_t + block_offset_d * stride_q_d
        k_ptr_t = k_ptr + k_offset + t * stride_k_t + block_offset_d * stride_k_d
        v_ptr_t = v_ptr + v_offset + t * stride_v_t + block_offset_d * stride_v_d
        h_ptr_t = h_ptr + h_offset + t * stride_h_t + block_offset_d * stride_h_d
        o_ptr_t = o_ptr + o_offset + t * stride_o_t + block_offset_d * stride_o_d
        s_ptr_t = s_ptr + s_offset + t * stride_s_t + block_offset_d * stride_s_d

        b_q = tl.load(q_ptr_t, mask=mask_d, other=0.0)
        b_k = tl.load(k_ptr_t, mask=mask_d, other=0.0)
        b_v = tl.load(v_ptr_t, mask=mask_d, other=0.0)
        b_h = tl.load(h_ptr_t, mask=mask_d, other=0.0)

        # Example decay factor
        d_i = 0.1

        # Accumulate partial sums
        b_s = b_q * b_k * d_i
        tl.store(s_ptr_t, b_s, mask=mask_d)

        # Compute final output
        b_o = (b_q * b_h) + (b_s * b_v)
        tl.store(o_ptr_t, b_o, mask=mask_d)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=1),
        triton.Config({'BLOCK_D': 32}, num_warps=2),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_bwd_kernel_dh(
    dh_ptr,
    k_ptr, v_ptr, h_ptr,
    B, T, D,
    stride_dh_b, stride_dh_t, stride_dh_d,
    stride_k_b, stride_k_t, stride_k_d,
    stride_v_b, stride_v_t, stride_v_d,
    stride_h_b, stride_h_t, stride_h_d,
    BLOCK_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    d_block_idx = tl.program_id(1)
    block_offset_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = block_offset_d < D

    dh_offset = batch_idx * stride_dh_b
    k_offset = batch_idx * stride_k_b
    v_offset = batch_idx * stride_v_b
    h_offset = batch_idx * stride_h_b

    # We'll accumulate the gradient of h in reverse time
    b_dh = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t in range(T - 1, -1, -1):
        dh_ptr_t = dh_ptr + dh_offset + t * stride_dh_t + block_offset_d * stride_dh_d
        k_ptr_t = k_ptr + k_offset + t * stride_k_t + block_offset_d * stride_k_d
        v_ptr_t = v_ptr + v_offset + t * stride_v_t + block_offset_d * stride_v_d
        h_ptr_t = h_ptr + h_offset + t * stride_h_t + block_offset_d * stride_h_d

        b_local_dh = tl.load(dh_ptr_t, mask=mask_d, other=0.0)
        b_k = tl.load(k_ptr_t, mask=mask_d, other=0.0)
        b_v = tl.load(v_ptr_t, mask=mask_d, other=0.0)
        b_h = tl.load(h_ptr_t, mask=mask_d, other=0.0)

        # Example decay factor
        d_b = 0.9
        d_i = 0.1

        # Combine existing grad with local grad
        b_local_dh += b_dh * d_b

        # Store updated gradient for h
        tl.store(dh_ptr_t, b_local_dh, mask=mask_d)

        # Update b_dh for next iteration
        # Dummy partial backprop for demonstration
        b_dh = b_local_dh * (b_k * b_v * d_i)


@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    dq_ptr, dk_ptr, dv_ptr,
    q_ptr, k_ptr, v_ptr, s_ptr, do_ptr,
    B, T, D,
    stride_dq_b, stride_dq_t, stride_dq_d,
    stride_dk_b, stride_dk_t, stride_dk_d,
    stride_dv_b, stride_dv_t, stride_dv_d,
    stride_q_b, stride_q_t, stride_q_d,
    stride_k_b, stride_k_t, stride_k_d,
    stride_v_b, stride_v_t, stride_v_d,
    stride_s_b, stride_s_t, stride_s_d,
    stride_do_b, stride_do_t, stride_do_d,
    BLOCK_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    d_block_idx = tl.program_id(1)
    block_offset_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = block_offset_d < D

    dq_offset = batch_idx * stride_dq_b
    dk_offset = batch_idx * stride_dk_b
    dv_offset = batch_idx * stride_dv_b
    q_offset = batch_idx * stride_q_b
    k_offset = batch_idx * stride_k_b
    v_offset = batch_idx * stride_v_b
    s_offset = batch_idx * stride_s_b
    do_offset = batch_idx * stride_do_b

    for t in range(T):
        dq_ptr_t = dq_ptr + dq_offset + t * stride_dq_t + block_offset_d * stride_dq_d
        dk_ptr_t = dk_ptr + dk_offset + t * stride_dk_t + block_offset_d * stride_dk_d
        dv_ptr_t = dv_ptr + dv_offset + t * stride_dv_t + block_offset_d * stride_dv_d
        q_ptr_t = q_ptr + q_offset + t * stride_q_t + block_offset_d * stride_q_d
        k_ptr_t = k_ptr + k_offset + t * stride_k
