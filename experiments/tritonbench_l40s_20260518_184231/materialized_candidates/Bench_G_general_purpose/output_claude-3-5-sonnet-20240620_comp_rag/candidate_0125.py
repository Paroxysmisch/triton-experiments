import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Tuple

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    # Pointers to matrices
    q, k, v, w, u,    # inputs
    o,                # output
    h0, ht,          # hidden states
    s_k_h, s_v_h,    # strides
    scale,           # scaling factor
    # Meta-parameters
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, 
    K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    # Program ID
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    
    # Compute pointer offsets
    offset = (T-1) * K if REVERSE else 0
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + offset
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + offset
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + offset
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + offset
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + offset

    # Compute masks
    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    # Initialize hidden state
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    # Load bonus term
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)

    # Main loop
    for _ in range(T):
        # Load inputs
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_w = tl.exp(tl.load(p_w, mask=mask_bk, other=0).to(tl.float32))
        
        # Compute key-value product
        b_kv = b_k[None, :] * b_v[:, None]
        
        # Compute output
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        
        # Update hidden state
        b_h = b_h * b_w[None, :] + b_kv
        
        # Store output
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        
        # Update pointers
        step = -K if REVERSE else K
        p_q += step
        p_k += step
        p_v += step
        p_w += step
        p_o += step

    # Store final hidden state if needed
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)
