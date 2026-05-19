import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    mask_flag,
    q_tile_ptr, k_tile_ptr, v_tile_ptr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Initialize pointers to Q, K, V
    q_tile_ptr = q_tile_ptr + off_hz * stride_qh
    k_tile_ptr = k_tile_ptr + off_hz * stride_kh
    v_tile_ptr = v_tile_ptr + off_hz * stride_vh
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q_ptrs = Q + q_tile_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k_ptrs = K + k_tile_ptr + (offs_n[None, :] + start_n) * stride_kn + offs_k[:, None] * stride_kk
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if mask_flag:
            mask = (start_n + offs_n) < N_CTX
            qk = tl.where(mask[None, :], qk, float("-inf"))
        qk *= sm_scale
        # -- compute scaling constant ---
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        l_curr = tl.exp(qk - m_curr[:, None])
        p = tl.exp(l_prev - m_prev)
        # -- update l and m and scale l ---
        l_prev = l_prev * p + tl.sum(l_curr, 1)
        m_prev = m_curr
        # -- scale l and update acc ---
        l_curr = l_curr / l_prev[:, None]
        acc = acc * l_curr[:, None]
        # -- update acc --
        v_ptrs = V + v_tile_ptr + (offs_n[None, :] + start_n) * stride_vn + offs_k[:, None] * stride_vk
        v = tl.load(v_ptrs)
        acc += tl.dot(l_curr, v)
    # write back l and m
    l_ptrs = L + q_tile_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(l_ptrs, m_prev[:, None] + tl.log(l_prev)[:, None])
    # initialize pointers to output
    out_ptrs = Out + q_tile_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(out_ptrs, acc)

@triton.jit
def _bwd_preprocess(
    Out, DO,
    L,
    Delta,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    off_hz,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)
    # compute scaling factor
    l = tl.load(L + off_hz * BLOCK_M + off_m[:, None] * BLOCK_DMODEL + off_d[None, :]).to(tl.float32)
    # scale and shift
    do = tl.load(DO + off_hz * BLOCK_M + off_m[:, None] * BLOCK_DMODEL + off_d[None, :])
    delta = tl.sum(do, axis=1) / BLOCK_DMODEL
    do = (do.to(tl.float32) - l[:, None] * delta[:, None]) / BLOCK_DMODEL
    # store
    dout_ptrs = Out + off_hz * BLOCK_M * BLOCK_DMODEL + off_m * BLOCK_DMODEL + off_d
    tl.store(dout_ptrs, do)
    # store delta
    delta_ptrs = Delta + off_hz * BLOCK_M + off_m
    tl.store(delta_ptrs, delta)

@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out,
    Delta, DO,
    L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    mask_flag,
    q_tile_ptr, k_tile_ptr, v_tile_ptr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers to Q, K, V
    q_tile_ptr = q_tile_ptr + off_hz * stride_qh
    k_tile_ptr = k_tile_ptr + off_hz * stride_kh
    v_tile_ptr = v_tile_ptr + off_hz * stride_vh
    # initialize pointer to m
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
