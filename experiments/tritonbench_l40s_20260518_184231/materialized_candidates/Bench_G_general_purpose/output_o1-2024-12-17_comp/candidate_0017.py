import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,
    q, q_scale,
    K_ptrs, K_scale_ptr,
    V_ptrs,
    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
    STAGE,
    offs_m, offs_n, N_CTX
):
    # -------------------------------------------------------
    # Load Q
    # Each thread program loads a portion of Q based on offs_m and HEAD_DIM
    q_vec = tl.load(q + (offs_m * HEAD_DIM), mask=offs_m < N_CTX)
    q_vec = q_vec * q_scale

    # -------------------------------------------------------
    # Stage 1 or 2 logic (softmax over block)
    # Stage 1: dot product, accumulate max for stable softmax
    # Stage 2: compute exp, weighted sum for values
    if STAGE == 1:
        # Initialize local max and sum
        tmp_max = tl.full([BLOCK_N], -float('inf'), tl.float32)
        tmp_sum = tl.zeros([BLOCK_N], tl.float32)

        # Load K, K-scale and compute attention score
        for n_block in range(0, BLOCK_N):
            k_vec = tl.load(K_ptrs[n_block] + (offs_m * HEAD_DIM), mask=offs_m < N_CTX)
            k_scale = tl.load(K_scale_ptr + n_block)
            att_score = tl.dot(q_vec, k_vec * k_scale)

            # Causal Mask: zero out invalid positions
            # Only valid if (start_m + offs_m) >= (n_block * BLOCK_N + offs_n)
            mask_cond = (start_m + offs_m) >= (n_block * BLOCK_N + offs_n)
            att_score = tl.where(mask_cond, att_score, float('-inf'))

            tmp_max = tl.maximum(tmp_max, att_score)
            tmp_sum += tl.exp(att_score)

        # Update m_i (max) and l_i (sum) buffers
        tl.store(m_i + offs_m, tmp_max)
        tl.store(l_i + offs_m, tmp_sum)
    else:
        # STAGE == 2
        # Compute normalized weights and accumulate into acc
        for n_block in range(0, BLOCK_N):
            # Reload attention scores from stage 1
            k_vec = tl.load(K_ptrs[n_block] + (offs_m * HEAD_DIM), mask=offs_m < N_CTX)
            k_scale = tl.load(K_scale_ptr + n_block)
            att_score = tl.dot(q_vec, k_vec * k_scale)

            # Causal Mask
            mask_cond = (start_m + offs_m) >= (n_block * BLOCK_N + offs_n)
            att_score = tl.where(mask_cond, att_score, float('-inf'))

            # Retrieve the local max from stage 1
            local_max = tl.load(m_i + offs_m)
            # Retrieve the local sum from stage 1
            local_sum = tl.load(l_i + offs_m)

            # Softmax
            att_score = tl.exp(att_score - local_max) / local_sum

            # Load V, accumulate weighted value in acc
            v_vec = tl.load(V_ptrs[n_block] + (offs_m * HEAD_DIM), mask=offs_m < N_CTX)
            wgt_v = v_vec * att_score
            acc_val = tl.load(acc + (offs_m * HEAD_DIM), mask=offs_m < N_CTX)
            tl.store(acc + (offs_m * HEAD_DIM), acc_val + wgt_v, mask=offs_m < N_CTX)


@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE,
    # metaparams
    **meta
):
    # Program ID for block-level scheduling
    bid_z = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_m = tl.program_id(2)

    # Start index for M-dim
    start_m = bid_m * BLOCK_M

    # Pointers and offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Advance Q, K, V, Out pointers to the current batch/head offset
    Q_ptr = Q + bid_z * stride_qz + bid_h * stride_qh + start_m * stride_qm
    K_ptr_base = K + bid_z * stride_kz + bid_h * stride_kh
    V_ptr_base = V + bid_z * stride_vz + bid_h * stride_vh
    O_ptr = Out + bid_z * stride_oz + bid_h * stride_oh + start_m * stride_om

    # Build pointer lists for K, V blocks
    K_ptrs = []
    V_ptrs = []
    for n_block in range(BLOCK_N):
        k_offset = n_block * stride_kn
        v_offset = n_block * stride_vk
        K_ptrs.append(K_ptr_base + k_offset)
        V_ptrs.append(V_ptr_base + v_offset)

    # K_scale pointer (one per block in N dim) if needed
    K_scale_ptr = K_scale + bid_z * H * BLOCK_N + bid_h * BLOCK_N

    # Create ephemeral buffers for accumulations
    acc = tl.zeros([BLOCK_M * HEAD_DIM], tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    m_i = tl.zeros([BLOCK_M], tl.float32)

    # Stage 1
    _attn_fwd_inner[1, BLOCK_M](
        acc, l_i, m_i,
        Q_ptr, Q_scale,
        K_ptrs, K_scale_ptr, V_ptrs,
        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 1,
        offs_m, offs_n, N_CTX
    )

    # Stage 2
    _attn_fwd_inner[1, BLOCK_M](
        acc, l_i, m_i,
        Q_ptr, Q_scale,
        K_ptrs, K_scale_ptr, V_ptrs,
        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 2,
        offs_m, offs_n, N_CTX
    )

    # Write final results from acc to Out
    for i in range(BLOCK_M):
        idx_m = start_m + i
        if idx_m < N_CTX:
            for hd in range(HEAD_DIM):
                out_val = tl.load(acc + i * HEAD_DIM + hd)
                tl.store(O_ptr + i * stride_om + hd * stride_on, out_val)


def forward(q, k, v, q_scale, k_scale):
    """
    Wrapper function that dispatches the Triton kernel for blockwise attention fwd pass.
    """
    # Shapes: q, k, v => [Z, H, N_CTX, HEAD_DIM]
    Z, H, N_CTX, HEAD_DIM = q.shape
    # Validate shapes
    assert k.shape == (Z, H, N_CTX, HEAD_DIM)
    assert v.shape == (Z, H, N_CTX, HEAD_DIM)

    # Output buffer
    import torch
    out = torch.empty_like(q)

    # Get strides
    stride_qz, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kz, stride_kh, stride_kn, stride_kk = k.stride()
    stride_vz, stride_vh, stride_vk, stride_vn = v.stride()
    stride_oz, stride_oh, stride_om, stride_on = out.stride()

    # BLOCK_M and BLOCK_N define block sizes
    BLOCK_M = 64
    BLOCK_N = 64

    # Launch grid
    #  - For each z in Z
    #  - For each h in H
    #  - For each block of M in range(0, N_CTX, BLOCK_M)
    grid = (Z, H, (N_CTX + BLOCK_M - 1) // BLOCK_M)

    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, 1,
        num_warps=4,
        num_stages=2
    )

    return out
