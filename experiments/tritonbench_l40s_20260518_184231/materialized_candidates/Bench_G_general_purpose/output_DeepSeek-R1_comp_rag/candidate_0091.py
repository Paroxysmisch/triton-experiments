import torch
import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    q, k, v, o,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_h = i_bh % H

    # Head-specific decay calculation
    decay_base = tl.math.pow(2, -5 - i_h * 1.0)
    b_b = tl.math.log2(1 - decay_base)

    # Initialize block pointers
    p_q = tl.make_block_ptr(
        q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
        (i_c * BTL, i_k * BK), (BTL, BK), (1, 0)
    )
    b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(q.dtype.element_ty)
    
    # Initialize output block
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    # Process non-overlapping blocks
    p_k = tl.make_block_ptr(
        k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
        (i_k * BK, 0), (BK, BTS), (0, 1)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
        (0, i_v * BV), (BTS, BV), (1, 0)
    )
    
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        decay = tl.math.exp2((BTS - tl.arange(0, BTS)) * b_b)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * decay[None, :]
        b_o = b_o * tl.math.exp2(b_b * BTS) + tl.dot(b_s.to(v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # Process overlapping blocks with masking
    o_q = tl.arange(0, BTL)
    p_k = tl.make_block_ptr(
        k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
        (i_k * BK, i_c * BTL), (BK, BTS), (0, 1)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
        (i_c * BTL, i_v * BV), (BTS, BV), (1, 0)
    )
    
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        o_k = tl.arange(0, BTS)
        mask = o_q[:, None] >= o_k[None, :]
        decay = tl.where(mask, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * decay
        b_o += tl.dot(b_s.to(v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    # Store final output
    p_o = tl.make_block_ptr(
        o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d),
        (i_c * BTL, i_v * BV), (BTL, BV), (1, 0)
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def parallel_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV
    i_h = i_bh % H

    # Gradient computation for queries
    p_do = tl.make_block_ptr(
        do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
        (i_c * BTL, i_v * BV), (BTL, BV), (1, 0)
    )
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    
    decay_base = tl.math.pow(2, -5 - i_h * 1.0)
    b_b = tl.math.log2(1 - decay_base)
    d_b = tl.math.exp2(b_b * BTS)

    # Compute dQ gradients
    p_k = tl.make_block_ptr(
        k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
        (0, i_k * BK), (BTS, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t),
        (i_v * BV, 0), (BV, BTS), (0, 1)
    )
    
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        decay = tl.math.exp2((BTS - tl.arange(0, BTS)) * b_b)
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * decay[None, :]
        b_dq = b_dq * d_b + tl.dot(b_ds.to(k.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
    
    # Compute dK and dV gradients
    p_q = tl.make_block_ptr(
        q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
        (i_k * BK, i_c * BTL), (BK, BTS), (0, 1)
    )
    p_do = tl.make_block_ptr(
        do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t),
        (i_v * BV, i_c * BTL), (BV, BTS), (0, 1)
    )
    
    b_dk = tl.zeros([BTL, BK], dtype=tl.float32)
    b_dv = tl.zeros([BTL, BV], dtype=tl.float32)
    o_q = tl.arange(0, BTS)
    
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        mask = o_q[:, None] <= tl.arange(0, BTL)[None, :]
        decay = tl.where(mask, tl.math.exp2((o_q[:, None] - tl.arange(0, BTL)[None, :]) * b_b), 0)
        
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False) * decay
        b_dk += tl.dot(b_s.to(q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dv += tl.dot(tl.trans(b_q), b_do, allow_tf32=False) * decay
        o_q += BTS

    # Store gradients
    p_dq = tl.make_block_ptr(
        dq + (i_bh + B * H * i_v) * s_qk_h, (T, K),
        (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0)
    )
    tl.store(p_dq, (b_dq * scale).to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    
    p_dk = tl.make_block_ptr(
        dk + (i_bh + B * H * i_v) * s_qk_h, (T, K),
        (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0)
    )
    tl.store(p_dk, (b_dk * scale).to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    
    p_dv = tl.make_block_ptr(
        dv + (i_bh + B * H * i_k) * s_vo_h, (T, V),
        (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0)
    )
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

class ParallelRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, T, K = q.shape
        V = v.shape[-1]
        
        # Configure kernel parameters
        BTL, BTS = 128, 32
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
        grid = (
            triton.cdiv(K, BK) * triton.cdiv(V, BV),
            triton.cdiv(T, BTL),
            B * H
        )
        
        # Allocate output tensor
        o = torch.empty((B, H, T, V), dtype=q.dtype, device=q.device)
        
        # Launch kernel
        parallel_retention_fwd_kernel[grid](
            q, k, v, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            K**-0.5,
            B=B, H=H, T=T, K=K, V=V,
            BTL=BTL, BTS=BTS, BK=BK, BV=BV,
            num_warps=4,
            num_stages=3
        )
        
        ctx.save_for_backward(q, k, v)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        
        # Configure backward kernel
        BTL, BTS = 64, 32
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
        grid = (
            triton.cdiv(K, BK) * triton.cdiv(V, BV),
            triton.cdiv(T, BTL),
            B * H
        )
        
        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch backward kernel
        parallel_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            K**-0.5,
            B=B, H=H, T=T, K=K, V=V,
            BTL=BTL, BTS=BTS, BK=BK, BV=BV,
            num_warps=4,
            num_stages=3
        )
        
        return dq, dk, dv

parallel_retention = ParallelRetentionFunction.apply
