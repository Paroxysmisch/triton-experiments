import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q, k, v, o, z,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale, use_normalize: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H

    # Load query block
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                           (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)) * scale

    # Initialize output and normalization
    b_o = tl.zeros((BTL, BV), dtype=tl.float32)
    b_z = tl.zeros((BTL,), dtype=tl.float32) if use_normalize else None

    # Process key/value blocks
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                           (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                           (0, i_v * BV), (BTS, BV), (1, 0))

    for _ in range(0, tl.cdiv(T, BTS)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # Compute attention scores
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        if use_normalize:
            b_z_block = tl.max(b_s, axis=1)  # Max for numerical stability
            b_s = tl.exp(b_s - b_z_block[:, None])
            b_z += b_s.sum(axis=1)
        
        # Accumulate output
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # Apply normalization if needed
    if use_normalize:
        b_o = b_o / b_z[:, None]
        p_z = tl.make_block_ptr(z + i_bh * T, (T,), (1,), (i_c * BTL,), (BTL,), (1,))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0,))

    # Store output
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                           (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _parallel_rebased_bwd_dq(
    q, k, v, do, dz, dq,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale, use_normalize: tl.constexpr,
    B, H, T, K, V,
    BTL, BTS, BK, BV,
    i_bh, i_c, i_k, i_v, i_h
):
    # Gradient computation for queries
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                           (i_k * BK, 0), (BK, T), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                           (0, i_v * BV), (T, BV), (1, 0))
    
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)
    
    # Compute gradient for Q
    b_dq = tl.dot(tl.dot(b_do, tl.trans(b_v)), tl.trans(b_k)) * scale
    
    if use_normalize:
        p_z = tl.make_block_ptr(dz + i_bh * T, (T,), (1,), (i_c * BTL,), (BTL,), (1,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_dq -= tl.sum(b_dq * b_z[:, None], axis=1)[:, None]
        b_dq /= b_z[:, None]
    
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                            (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _parallel_rebased_bwd_dkv(
    q, k, v, do, dz, dk, dv,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale, use_normalize: tl.constexpr,
    B, H, T, K, V,
    BTL, BTS, BK, BV,
    i_bh, i_c, i_k, i_v, i_h
):
    # Gradient computation for keys/values
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
                          (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    
    # Compute gradients for K and V
    b_dk = tl.dot(tl.trans(b_q), tl.dot(b_do, tl.trans(v))) * scale
    b_dv = tl.dot(tl.trans(b_q), tl.dot(k, b_do)) * scale
    
    if use_normalize:
        p_z = tl.make_block_ptr(dz + i_bh * T, (T,), (1,), (i_c * BTL,), (BTL,), (1,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_dk -= tl.sum(b_dk * b_z[None, :], axis=0)[None, :]
        b_dv -= tl.sum(b_dv * b_z[None, :], axis=0)[None, :]
        b_dk /= b_z[None, :]
        b_dv /= b_z[None, :]
    
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t),
                            (i_k * BK, i_c * BTL), (BK, BTL), (0, 1))
    p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def parallel_rebased_bwd_kernel(
    q, k, v, do, dz, dq, dk, dv,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    scale, use_normalize: tl.constexpr,
    B, H, T, K, V,
    BTL, BTS, BK, BV
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    
    _parallel_rebased_bwd_dq(
        q, k, v, do, dz, dq,
        s_qk_h, s_qk_t, s_qk_d,
        s_vo_h, s_vo_t, s_vo_d,
        scale, use_normalize,
        B, H, T, K, V,
        BTL, BTS, BK, BV,
        i_bh, i_c, i_k, i_v, i_h
    )
    
    _parallel_rebased_bwd_dkv(
        q, k, v, do, dz, dk, dv,
        s_qk_h, s_qk_t, s_qk_d,
        s_vo_h, s_vo_t, s_vo_d,
        scale, use_normalize,
        B, H, T, K, V,
        BTL, BTS, BK, BV,
        i_bh, i_c, i_k, i_v, i_h
    )

class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, use_scale=True, use_normalize=False):
        B, H, T, K = q.shape
        V = v.shape[-1]
        assert K <= 128 and V <= 128, "Feature dim exceeds 128"
        
        # Initialize outputs
        o = torch.empty_like(v)
        z = torch.empty(B, H, T, device=q.device) if use_normalize else None
        
        # Kernel configuration
        BTL, BTS = 64, 32
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
        grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV), triton.cdiv(T, BTL), B * H)
        
        scale = K ** -0.5 if use_scale else 1.0
        
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            scale, use_normalize,
            B, H, T, K, V,
            BTL, BTS, BK, BV
        )
        
        ctx.save_for_backward(q, k, v, z)
        ctx.use_scale = use_scale
        ctx.use_normalize = use_normalize
        return o if not use_normalize else (o, z)

    @staticmethod
    def backward(ctx, do, dz=None):
        q, k, v, z = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        
        # Initialize gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Kernel configuration
        BTL, BTS = 64, 32
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
        grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV), triton.cdiv(T, BTL), B * H)
        
        scale = K ** -0.5 if ctx.use_scale else 1.0
        
        parallel_rebased_bwd_kernel[grid](
            q, k, v, do, dz if ctx.use_normalize else None,
            dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            scale, ctx.use_normalize,
            B, H, T, K, V,
            BTL, BTS, BK, BV
        )
        
        return dq, dk, dv, None, None

def parallel_rebased(q, k, v, use_scale=True, use_normalize=False, return_both=False):
    output = ParallelBasedFunction.apply(q, k, v, use_scale, use_normalize)
    if return_both and use_normalize:
        return output[0], output[1]
    return output if not use_normalize else output[0]
