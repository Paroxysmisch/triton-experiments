import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q, k, v, w, u, o, h0, ht,
    s_k_h, s_v_h, scale,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    # Initialize pointers with reverse handling
    p_q = q + i_bh * s_k_h + i_k * BK + ((T-1)*K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + ((T-1)*K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + ((T-1)*V if REVERSE else 0)
    p_o = o + (i_bh + i_k*B*H) * s_v_h + i_v * BV + ((T-1)*V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + ((T-1)*K if REVERSE else 0)
    p_u = u + i_h*K + i_k*BK

    # Masking for boundary checks
    mask_bk = (i_k*BK + tl.arange(0, BK)) < K
    mask_bv = (i_v*BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh*K*V + (i_k*BK + tl.arange(0, BK)[None, :])*V + (i_v*BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0)

    # Load bonus vector
    b_u = tl.load(p_u, mask=mask_bk, other=0)

    # Main temporal loop
    for _ in range(T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_w = tl.exp(tl.load(p_w, mask=mask_bk, other=0))

        # Compute output and update state
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        tl.store(p_o, tl.sum(b_o, axis=1).to(p_o.dtype.element_ty), mask=mask_bv)
        
        b_h = b_h * b_w[None, :] + b_kv

        # Update pointers with direction handling
        step = -1 if REVERSE else 1
        p_q += step * K
        p_k += step * K
        p_v += step * V
        p_o += step * V
        p_w += step * K

    # Store final state if needed
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh*K*V + (i_k*BK + tl.arange(0, BK)[None, :])*V + (i_v*BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)
