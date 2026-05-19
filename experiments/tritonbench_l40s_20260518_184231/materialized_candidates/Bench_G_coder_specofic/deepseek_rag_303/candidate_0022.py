import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, K_cache, V_cache,
    Out, DO,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk, 
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    START_M, m_group_size_m, N_CTX, m_group_size_n, HR,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STAGE: tl.constexpr
):
    if not STAGE: return
    if STAGE == 1:
        off_hz = tl.program_id(1)
        off_z = off_hz // HR
        off_h = off_hz % HR

        qk_offset = off_z * stride_qz + off_h * stride_qh
        kv_offset = qk_offset // stride_qm
        offs_n = tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M) + START_M * BLOCK_M
        Q_ptrs = (Q + qk_offset +
                  offs_m[:, None] * stride_qm +
                  tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk)
        K_cache_ptrs = K_cache + qk_offset + \
            offs_n[None, :] * stride_kn + \
            tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kk
        K_mask = offs_n[None, :] < N_CTX
        if USE_INITIAL_STATE:
            vs = tl.load(V_cache + kv_offset + offs_n * stride_vn +
                         tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk)
            K_0 = tl.load(K_cache_ptrs, mask=K_mask)
            K_cache_ptrs += tl.cdiv(N_CTX, BLOCK_N) * stride_kn
            V_ptrs = V_cache + kv_offset + \
                offs_n[:, None] * stride_qm + \
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
            tl.store(V_cache + (kv_offset +
                               offs_n * stride_vn +
                               tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk),
                     vs, mask=offs_n[None, :] < m_group_size_n)
            tl.store(K_cache_ptrs, K_0, mask=K_mask)
        else:
            V_ptrs = (V + kv_offset +
                      offs_n[:, None] * stride_qm +
                      tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk)
        K_ptrs = (K + kv_offset +
                  offs_n[None, :] * stride_kn +
                  tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kk)
    elif STAGE == 2:
        off_hz = tl.program_id(1)
        off_z = off_hz // HR
        off_h = off_hz % HR

        qk_offset = off_z * stride_qz + stride_qh * off_h
        kv_offset = qk_offset // stride_qm
        offs_m = tl.arange(0, BLOCK_M) + START_M * BLOCK_M
        offs_n = tl.arange(0, BLOCK_N)
        mask1 = offs_m[:, None] < N_CTX
        Q_ptrs = Q + qk_offset + \
            offs_m[:, None] * stride_qm + \
            tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        V_ptrs = V_cache + (kv_offset +
                            tl.arange(0, BLOCK_DMODEL)[
                                None, :] * stride_qk +
                            offs_n[:, None] * stride_qm)
        K_cache_ptrs = (K_cache + (kv_offset +
                                   tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk +
                                   offs_m[:, None] * stride_qm))
        mask = offs_n[None, :] < N_CTX
        if IS_CAUSAL:
            mask &= (offs_n[:, None] >= (offs_m[:, None] - START_M * BLOCK_M + BLOCK_M - 1))
        K_mask = mask
    else:
        off_hz = tl.program_id(1)
        off_z = off_hz // HR
        off_h = off_hz % HR

        qk_offset = (off_z + 1) * stride_qz + off_h * stride_qh
        kv_offset = qk_offset // stride_qm
        offs_m = tl.arange(0, BLOCK_M) + START_M * BLOCK_M
        offs_n = tl.arange(0, BLOCK_N)
        Q_ptrs = (Q + qk_offset +
                  offs_m[:, None] * stride_qm +
                  tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk)
        V_cache_ptrs = V_cache + qk_offset + \
            offs_n[:, None] * stride_qm + \
            tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        V_ptrs = V_cache + kv_offset + (offs_n + tl.cdiv(N_CTX, BLOCK_N) *
                                        BLOCK_N)[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        K_cache_ptrs = K_cache + kv_offset + (offs_m + tl.cdiv(N_CTX, BLOCK_M) *
                                              BLOCK_M)[None, :] * stride_qm + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_qk
        K_mask = offs_m[None, :] < N_CTX

    if STAGE == 1 or STAGE == 2:
        q = tl.load(Q_ptrs, mask=mask1)
        q = (q * sm_scale).to(tl.float32)

    Initial_sum = 0.
    Acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    V_inc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    if STAGE != 3:
        block_counter = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - float("inf")

    if STAGE == 1 or STAGE == 3:
        for start_n in range(0, (offs_n.shape[0] + BLOCK_N - 1) // B
