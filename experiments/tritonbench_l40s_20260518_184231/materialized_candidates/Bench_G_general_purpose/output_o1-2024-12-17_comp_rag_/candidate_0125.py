import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from fla.ops.utils import chunk_reversed_cumsum_fwd

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q,  # [B, H, T, K]
    k,  # [B, H, T, K]
    v,  # [B, H, T, V]
    w,  # [B, H, T, K]
    u,  # [B, H, K]
    o,  # [NK, B, H, T, V]
    h0, # [B, H, K, V]
    ht, # [B, H, K, V]
    s_k_h, s_v_h, scale,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_h = i_bh % H

    # Offsets
    # For q, k, w we load sub-block BK, for v and o sub-block BV
    # For reversed sequence, we start from T-1 and go backwards
    p_q = q + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1) * K if REVERSE else 0 )
    p_k = k + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1) * K if REVERSE else 0 )
    p_v = v + (i_bh * s_v_h) + (i_v * BV) + tl.arange(0, BV) + ( (T-1) * V if REVERSE else 0 )
    # o has shape [NK, B, H, T, V], so we need to compute appropriate strides:
    #   each sub-block in dimension T changes by V
    #   each sub-block in dimension B/H changes by s_v_h
    #   each sub-block in dimension NK changes by ?
    #
    # We'll keep it simpler by flattening (NK, B, H) together in store.
    # index = i_bh + i_k * (B*H)
    # so o += index * block in dim 0
    # but the stride for T is still V, so we do T-1 * V if reversed
    p_o = o + ( (i_bh + i_k * B * H) * (s_v_h) ) + (i_v * BV) + tl.arange(0, BV) + ( (T-1) * V if REVERSE else 0 )

    p_w = w + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1) * K if REVERSE else 0 )
    # bonus u has shape [B, H, K], but when i_k !=0, we shift
    p_u = u + (i_h * K) + (i_k * BK) + tl.arange(0, BK)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    # For hidden state, shape [BV, BK], so we can broadcast
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    # load initial hidden state if needed
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # h0 has shape [B, H, K, V], we load sub-block [BK, BV]
        # offset: i_bh indexes B*H, then i_k * BK in K dimension, i_v * BV in V dimension
        p_h0 = h0 + (i_bh * K * V) + ( (i_k * BK + tl.arange(0, BK)[None, :]) * V ) + ( (i_v * BV + tl.arange(0, BV)[:, None]) )
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)

    # main recurrent loop
    # at each step: read k, v, w, q, update state => compute o
    for _ in range(T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        # b_kv is [BV, BK]
        b_kv = b_k[None, :] * b_v[:, None]

        # output: (b_h + b_kv * b_u[None, :]) * b_q[None, :] => [BV, BK], sum across BK => [BV]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)  # -> [BV]

        # update hidden state
        b_h = b_h * b_w[None, :]
        b_h += b_kv

        # store output
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)

        # shift pointers
        step = -1 if REVERSE else 1
        p_q += step * K
        p_k += step * K
        p_o += step * V
        p_v += step * V
        p_w += step * K

    # store final hidden state
    if STORE_FINAL_STATE:
        p_ht = ht + (i_bh * K * V) + ( (i_k * BK + tl.arange(0, BK)[None, :]) * V ) + ( (i_v * BV + tl.arange(0, BV)[:, None]) )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dq(
    k, v, w, u, do, dq, dq_aux, h0,
    s_k_h, s_v_h, scale,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_h = i_bh % H

    # pointers
    p_k = k + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if REVERSE else 0 )
    p_v = v + (i_bh * s_v_h) + (i_v * BV) + tl.arange(0, BV) + ( (T-1)*V if REVERSE else 0 )
    p_do = do + (i_bh * s_v_h) + (i_v * BV) + tl.arange(0, BV) + ( (T-1)*V if REVERSE else 0 )
    # dq, dq_aux have shape [NV, B, H, T, K], but we flatten i_bh + i_v*(B*H) as well
    p_dq = dq + ( (i_bh + i_v * B * H) * s_k_h ) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if REVERSE else 0 )
    p_dq_aux = dq_aux + ( (i_bh + i_v * B * H) * s_k_h ) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if REVERSE else 0 )
    p_w = w + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if REVERSE else 0 )
    p_u = u + (i_h * K) + (i_k * BK) + tl.arange(0, BK)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + (i_bh * K * V) + ( (i_k * BK + tl.arange(0, BK)[None, :]) * V ) + ( (i_v * BV + tl.arange(0, BV)[:, None]) )
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_kv = b_k[None, :] * b_v[:, None]
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)

        # partial for q
        h_q = b_h * b_do[:, None]  # shape [BV, BK]
        b_dq = tl.sum(h_q + b_kv * b_u[None, :] * b_do[:, None], axis=0)
        b_dq *= scale
        b_dq_aux = tl.sum(h_q, axis=0)

        # update hidden
        b_h = b_h * b_w[None, :]
        b_h += b_kv

        # store partial for q
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_bk)
        tl.store(p_dq_aux, b_dq_aux.to(p_dq_aux.dtype.element_ty), mask=mask_bk)

        step = -1 if REVERSE else 1
        p_k += step * K
        p_do += step * V
        p_v += step * V
        p_w += step * K
        p_dq += step * K
        p_dq_aux += step * K

@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dkv(
    q, k, v, w, u, do, dk, dk_aux, dv, dh0,
    s_k_h, s_v_h, scale, B, H, T,
    BK: tl.constexpr, BV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_h = i_bh % H

    # for q, k, w we move forward if REVERSE else backward in memory
    # we read from T-1 -> 0 if not REVERSE, else 0 -> T-1
    p_q = q + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if not REVERSE else 0 )
    p_k = k + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if not REVERSE else 0 )
    p_do = do + (i_bh * s_v_h) + (i_v * BV) + tl.arange(0, BV) + ( (T-1)*V if not REVERSE else 0 )
    p_v = v + (i_bh * s_v_h) + (i_v * BV) + tl.arange(0, BV) + ( (T-1)*V if not REVERSE else 0 )
    p_dk = dk + ( (i_bh + i_v * B * H ) * s_k_h ) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if not REVERSE else 0 )
    p_dk_aux = dk_aux + ( (i_bh + i_v * B * H ) * s_k_h ) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if not REVERSE else 0 )
    # dv has shape [NK, B, H, T, V], flatten i_bh + i_k*(B*H)
    p_dv = dv + ( (i_bh + i_k * B * H ) * s_v_h ) + (i_v * BV) + tl.arange(0, BV) + ( (T-1)*V if not REVERSE else 0 )
    p_w = w + (i_bh * s_k_h) + (i_k * BK) + tl.arange(0, BK) + ( (T-1)*K if not REVERSE else 0 )

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    p_u = u + (i_h * K) + (i_k * BK) + tl.arange(0, BK)
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)

    # loop backward if not REVERSE, else forward
    # each iteration we compute partial derivatives for k, v
    for _ in range(T-1, -1, -1):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        b_dkv = b_q[:, None] * b_do[None, :]  # shape [BK, BV]

        # partial for k
        b_dk_ = tl.sum( b_dh * b_v[None, :], axis=1 )
        tl.store(p_dk_aux, b_dk_.to(p_dk_aux.dtype.element_ty), mask=mask_bk)

        b_dk_ += tl.sum( b_dkv * b_u[:, None] * b_v[None, :], axis=1 )

        # partial for v
        b_dv_ = tl.sum( (b_dh + (b_dkv * b_u[:, None])) * b_k[:, None], axis=0 )

        # store
        tl.store(p_dk, b_dk_.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, b_dv_.to(p_dv.dtype.element_ty), mask=mask_bv)

        # update hidden
        b_dh *= tl.exp(b_w)[:, None]
        b_dh +=
