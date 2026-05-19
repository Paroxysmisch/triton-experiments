import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q_PTR, K_PTR, V_PTR,
    O_PTR, M_PTR, D_PTR,
    stride_qbs, stride_qhs, stride_qds,
    stride_kbs, stride_khs, stride_kds,
    stride_vbs, stride_vhs, stride_vds,
    stride_obs, stride_ohs, stride_ods,
    stride_m, stride_d,
    B, H, N_CTX, D_HEAD,
    sm_scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Current block for query
    off_m = pid_m * BLOCK_M
    # Current batch-head
    b = pid_bh // H
    h = pid_bh % H

    # Offsets for Q
    q_ptrs = Q_PTR + b * stride_qbs + h * stride_qhs + off_m * stride_qds

    # Initialize running max for partial softmax
    max_val = tl.full([BLOCK_M], float('-inf'), tl.float32)
    # Initialize running sum of exp
    exp_sum = tl.zeros([BLOCK_M], tl.float32)

    # Create range of offsets
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = off_m + offs_m
    # Load query
    q = tl.load(q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qds + tl.arange(0, D_HEAD) * 1, mask=mask_m < N_CTX, other=0.0)

    # Iterate over K/V in steps of BLOCK_N
    n_steps = (N_CTX + BLOCK_N - 1) // BLOCK_N
    for step in range(n_steps):
        off_n = step * BLOCK_N
        # Load K/V block
        k_ptrs = K_PTR + b * stride_kbs + h * stride_khs + off_n * stride_kds
        v_ptrs = V_PTR + b * stride_vbs + h * stride_vhs + off_n * stride_vds
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = off_n + offs_n

        # [BLOCK_N, D_HEAD]
        k = tl.load(k_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_kds + tl.arange(0, D_HEAD) * 1, mask=mask_n < N_CTX, other=0.0)

        # Dot product [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Causal mask
        if CAUSAL:
            # We only mask if off_n <= off_m, else all is valid
            row_mask = (mask_m[:, None] < (off_n + offs_n[None, :]))
            qk = tl.where(row_mask, qk, float('-inf'))

        # Compute max
        block_max = tl.maximum(tl.max(qk, 1), max_val)
        # Adjust old exp_sum with new scaling factor
        exp_sum_scale = tl.exp2(max_val - block_max)
        exp_sum *= exp_sum_scale
        # Update new max_val
        max_val = block_max
        # Accumulate
        exp_qk = tl.exp2(qk - max_val[:, None])
        exp_sum = exp_sum + tl.sum(exp_qk, 1)

        # Write partial max and denominator
        tl.store(M_PTR + off_m + offs_m, max_val, mask=mask_m < N_CTX)
        tl.store(D_PTR + off_m + offs_m, exp_sum, mask=mask_m < N_CTX)

        # Multiply exp_qk by V
        v = tl.load(v_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_vds + tl.arange(0, D_HEAD) * 1, mask=mask_n < N_CTX, other=0.0)
        out = tl.dot(exp_qk, v)
        # Write partial to output
        o_ptrs = O_PTR + b * stride_obs + h * stride_ohs + off_m * stride_ods
        old_o = tl.load(o_ptrs + offs_m[:, None] * stride_ods + tl.arange(0, D_HEAD) * 1, mask=mask_m < N_CTX, other=0.0)
        out += old_o
        tl.store(o_ptrs + offs_m[:, None] * stride_ods + tl.arange(0, D_HEAD) * 1, out, mask=mask_m < N_CTX)


@triton.jit
def _normalize_kernel(
    O_PTR, M_PTR, D_PTR,
    stride_obs, stride_ohs, stride_ods,
    stride_m, stride_d,
    B, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_m = pid_m * BLOCK_M
    b = pid_bh // H
    h = pid_bh % H

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = off_m + offs_m
    # Load max and denom
    m_val = tl.load(M_PTR + off_m + offs_m, mask=mask_m < N_CTX)
    d_val = tl.load(D_PTR + off_m + offs_m, mask=mask_m < N_CTX)

    # Normalization
    denom = 1.0 / d_val
    o_ptrs = O_PTR + b * stride_obs + h * stride_ohs + off_m * stride_ods
    out = tl.load(
        o_ptrs + offs_m[:, None] * stride_ods + tl.arange(0, D_HEAD),
        mask=mask_m < N_CTX,
        other=0.0
    )
    out *= denom[:, None]
    tl.store(
        o_ptrs + offs_m[:, None] * stride_ods + tl.arange(0, D_HEAD),
        out,
        mask=mask_m < N_CTX
    )


def flash_attn_triton(q, k, v, sm_scale, causal=False):
    B, H, N_CTX, D_HEAD = q.shape
    # Create output
    import torch
    o = torch.zeros_like(q)
    # Buffers for max and denominator
    m = torch.empty((B * H, N_CTX), dtype=torch.float32, device=q.device)
    d = torch.empty((B * H, N_CTX), dtype=torch.float32, device=q.device)

    grid = lambda META: (
        ( (N_CTX + META['BLOCK_M'] - 1) // META['BLOCK_M'], B * H ),
    )

    # Kernel for partial forward
    _fwd_kernel[grid](
        q, k, v, o, m, d,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        m.stride(0), d.stride(0),
        B, H, N_CTX, D_HEAD,
        sm_scale,
        CAUSAL=causal,
        BLOCK_M=64,  # typical block settings
        BLOCK_N=64
    )

    # Kernel to normalize
    _normalize_kernel[grid](
        o, m, d,
        o.stride(0), o.stride(1), o.stride(2),
        m.stride(0), d.stride(0),
        B, H, N_CTX, D_HEAD,
        BLOCK_M=64
    )

    return o
