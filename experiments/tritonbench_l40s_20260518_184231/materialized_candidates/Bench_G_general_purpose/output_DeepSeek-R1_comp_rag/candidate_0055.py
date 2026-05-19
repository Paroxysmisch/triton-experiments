import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

# Forward Kernels and Wrappers
@triton.jit
def chunk_retention_fwd_kernel_h(
    # Pointers to tensors
    k, v, h, initial_state, final_state,
    # Tensor strides
    s_kh, s_kt, s_kd,
    s_vh, s_vt, s_vd,
    s_hh, s_ht, s_hd,
    # Shape parameters
    B, H, T, scale,
    # Hyperparameters
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(BT * b_b)
    d_h = tl.math.exp2((BT - o_i - 1) * b_b)

    p_k = tl.make_block_ptr(k + i_bh * s_kh, (T, DK), (s_kt, s_kd), (0, i_k*BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vh, (T, DV), (s_vt, s_vd), (0, i_v*BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (T, DK, DV), (s_ht, s_hd, 1), (0, i_k*BK, i_v*BV), (BT, BK, BV), (1, 1, 1))

    current_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_init = tl.make_block_ptr(initial_state + i_bh * DK*DV, (DK, DV), (DV, 1), (i_k*BK, i_v*BV), (BK, BV), (1, 0))
        current_h = tl.load(p_init, boundary_check=(0,1))

    for i in range(tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0,1))
        b_v = tl.load(p_v, boundary_check=(0,1))
        
        current_h = d_b * current_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)
        tl.store(p_h, current_h.to(p_h.dtype.element_ty), boundary_check=(0,1,2))
        
        p_k = tl.advance(p_k, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (BT, 0, 0))

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh*DK*DV, (DK, DV), (DV, 1), (i_k*BK, i_v*BV), (BK, BV), (1, 0))
        tl.store(p_final, current_h.to(p_final.dtype.element_ty), boundary_check=(0,1))

@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o,
    s_qh, s_qt, s_qd,
    s_kh, s_kt, s_kd,
    s_vh, s_vt, s_vd,
    s_oh, s_ot, s_od,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_o = tl.math.exp2((o_i + 1) * b_b)

    p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, DK), (s_qt, s_qd), (0, i_k*BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_kh, (DK, T), (s_kd, s_kt), (i_k*BK, 0), (BK, BT), (0, 1))
    p_h = tl.make_block_ptr(h + i_bh * DK*DV, (T, DK, DV), (DK*DV, DV, 1), (0, i_k*BK, i_v*BV), (BT, BK, BV), (1, 1, 1))
    p_o = tl.make_block_ptr(o + i_bh * s_oh, (T, DV), (s_ot, s_od), (0, i_v*BV), (BT, BV), (1, 0))

    for i in range(tl.cdiv(T, BT)):
        b_q = tl.load(p_q, boundary_check=(0,1)) * scale
        b_k = tl.load(p_k, boundary_check=(0,1))
        b_h = tl.load(p_h, boundary_check=(0,1,2))
        
        s = tl.dot(b_q, b_k, allow_tf32=False)
        o_part = tl.dot(s.to(b_q.dtype), tl.load(p_v), allow_tf32=False)
        o_h = tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False) * d_o[:, None]
        tl.store(p_o, (o_part + o_h).to(p_o.dtype.element_ty), boundary_check=(0,1))
        
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_h = tl.advance(p_h, (BT, 0, 0))
        p_o = tl.advance(p_o, (BT, 0))

# Backward Kernels and Wrappers
@triton.jit
def chunk_retention_bwd_kernel_dh(
    q, do, dh,
    s_qh, s_qt, s_qd,
    s_oh, s_ot, s_od,
    s_dhh, s_dht, s_dhd,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(BT * b_b)
    d_q = tl.math.exp2((o_i + 1) * b_b) * scale

    p_q = tl.make_block_ptr(q + i_bh * s_qh, (DK, T), (s_qd, s_qt), (i_k*BK, T-BT), (BK, BT), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_oh, (T, DV), (s_ot, s_od), (T-BT, i_v*BV), (BT, BV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_dhh, (T, DK, DV), (s_dht, s_dhd, 1), (T-BT, i_k*BK, i_v*BV), (BT, BK, BV), (1, 1, 1))

    dh_accum = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(tl.cdiv(T, BT)):
        b_q = tl.load(p_q, boundary_check=(0,1))
        b_do = tl.load(p_do, boundary_check=(0,1))
        
        dh_accum = d_b * dh_accum + tl.dot(b_q, (b_do * d_q[:, None]).to(b_q.dtype), allow_tf32=False)
        tl.store(p_dh, dh_accum.to(p_dh.dtype.element_ty), boundary_check=(0,1,2))
        
        p_q = tl.advance(p_q, (0, -BT))
        p_do = tl.advance(p_do, (-BT, 0))
        p_dh = tl.advance(p_dh, (-BT, 0, 0))

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q, k, v, h, do, dh, dq, dk, dv,
    s_qh, s_qt, s_qd,
    s_kh, s_kt, s_kd,
    s_vh, s_vt, s_vd,
    s_oh, s_ot, s_od,
    s_dqh, s_dqt, s_dqd,
    s_dkh, s_dkt, s_dkd,
    s_dvh, s_dvt, s_dvd,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    
    # Implementation similar to fused version but split for dq, dk, dv
    # ... (detailed kernel implementation for gradient calculations)

# Wrapper Functions
def chunk_fwd_h_fn(q, k, v, h, initial_state, final_state, BT=64):
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    BK, BV = min(64, triton.next_power_of_2(DK)), min(64, triton.next_power_of_2(DV))
    grid = (triton.cdiv(DV, BV), triton.cdiv(DK, BK), B*H)
    
    chunk_retention_fwd_kernel_h[grid](
        k, v, h, initial_state, final_state,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        B, H, T, DK**-0.5,
        BT=BT, DK=DK, DV=DV, BK=BK, BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        CHECK=False,
        num_warps=4,
        num_stages=1
    )

def chunk_fwd_o_fn(q, k, v, h, o, BT=64):
    # Similar configuration for output kernel
    # ... (implementation details)

def chunk_bwd_dh_fn(q, do, dh, BT=64):
    # Backward hidden state gradient configuration
    # ... (implementation details)

def chunk_bwd_dqkv_fn(q, k, v, h, do, dh, dq, dk, dv, BT=64):
    # Backward input gradients configuration
    # ... (implementation details)

# Autograd Function
class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, initial_state, output_final_state):
        # Forward pass using chunk_fwd_h_fn and chunk_fwd_o_fn
        # ... (implementation similar to fused version but split)
    
    @staticmethod
    @custom_bwd
    def backward(ctx, do, d_final_state):
        # Backward pass using chunk_bwd_dh_fn and chunk_bwd_dqkv_fn
        # ... (implementation details)

def chunk_retention(q, k, v, initial_state=None, output_final_state=False):
    # User-friendly wrapper
    return ChunkRetentionFunction.apply(q, k, v, initial_state, output_final_state)
