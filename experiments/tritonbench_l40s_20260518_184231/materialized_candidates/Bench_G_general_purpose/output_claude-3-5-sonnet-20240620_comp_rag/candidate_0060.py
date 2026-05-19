import torch
import triton
import triton.language as tl

@triton.jit
def fused_chunk_gla_fwd_kernel(
    q, k, v, g, o,
    initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        d_b = tl.load(p_db).to(tl.float32)
        if CHECK and i == 0:
            b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
            b_h = b_h * tl.math.exp2(d_b)[:, None] + tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)
        else:
            b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
            b_h = b_h * tl.math.exp2(d_b)[:, None] + tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_db += BT * DK

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def fused_chunk_gla_bwd_kernel(
    q, k, v, g,
    do, dq, dk, dv,
    initial_state,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + ((i+1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh+i_v*B*H)*s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        d_b = tl.load(p_db).to(tl.float32)

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h * tl.math.exp2(d_b)[None, :] + tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h * tl.math.exp2(d_b)[None, :] + tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    b_h = None
    tl.debug_barrier()
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + (T - (i-1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK),
                                 (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV),
                                 (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_db = tl.load(p_db).to(tl.float32)

        if CHECK and i == 1:
            b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False)
            b_dh = b_dh * tl.math.exp2(b_db)[:, None] + tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)
        else:
            b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False)
            b_dh = b_dh * tl.math.exp2(b_db)[:, None] + tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

class FusedChunkGLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, scale, initial_state=None, output_final_state=False):
        B, H, T, DK = q.shape
        DV = v.shape[-1]
        dtype = q.dtype
        device = q.device
        
        o = torch.empty_like(v)
        final_state = torch.empty((B, H, DK, DV), dtype=dtype, device=device) if output_final_state else None
        
        BLOCK_MODEL_K = min(triton.next_power_of_2(DK), 64)
        BLOCK_MODEL_V = min(triton.next_power_of_2(DV), 64)
        BLOCK_T = 128
        
        grid = (triton.cdiv(DV, BLOCK_MODEL_V), triton.cdiv(DK, BLOCK_MODEL_K), B * H)
        
        fused_chunk_gla_fwd_kernel[grid](
            q, k, v, g, o,
            initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BLOCK_T=BLOCK_T, BK=BLOCK_MODEL_K, BV=BLOCK_MODEL_V,
            DK=DK, DV=DV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            CHECK=True,
            num_warps=4,
            num_stages=1,
        )
        
        ctx.save_for_backward(q, k, v, g, initial_state)
        ctx.scale = scale
        
        if output_final_state:
            return o, final_state
        return o

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, g, initial_state = ctx.saved_tensors
        scale = ctx.scale
        
        B, H, T, DK = q.shape
        DV = v.shape[-1]
        dtype = q.dtype
        device = q.device
        
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        BLOCK_MODEL_K = min(triton.next_power_of_2(DK), 64)
        BLOCK_MODEL_V = min(triton.next_power_of_2(DV), 64)
        BLOCK_T = 128
        
        grid = (triton.cdiv(DV, BLOCK_MODEL_V), triton.cdiv(DK, BLOCK_MODEL_K), B * H)
        
        fused_chunk_gla_bwd_kernel[grid](
            q, k, v, g,
            do, dq, dk, dv,
            initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BLOCK_T=BLOCK_T, BK=BLOCK_MODEL_K, BV=BLOCK_MODEL_V,
            DK=DK, DV=DV,
            USE_INITIAL_STATE=initial_state is not None,
            CHECK=True,
            num_warps=4,
            num_stages=1,
        )
        
        return dq, dk, dv, None, None, None, None

def fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
):
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    seq_len = v.shape[-2]
    d_head_v = v.shape[-1]
    q, k, v, g = map(lambda x: pad_to_multiple(x, 128), [q, k, v, g])
    o, final_state = FusedChunkGLAFunction.apply(
        q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :d_head_v]
    if output_final_state:
        return o, final_state
    return o

def pad_to_multiple(x: torch.Tensor, multiple: int):
    seq_len = x.shape[-2]
    padded_seq_len = (seq_len + multiple - 1) // multiple * multiple
    return torch.nn.functional.pad(x, (0, 0, 0, padded_seq_len - seq_len))
