import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, w_ptr, u_ptr,
    o_ptr,
    initial_state_ptr, final_state_ptr,
    # Matrix dimensions
    B, H, T, K, V,
    # Strides for q
    stride_q_b, stride_q_h, stride_q_t, stride_q_k,
    # Strides for k
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    # Strides for v
    stride_v_b, stride_v_h, stride_v_t, stride_v_v,
    # Strides for w
    stride_w_h, stride_w_k,
    # Strides for u
    stride_u_h, stride_u_v,
    # Strides for o
    stride_o_b, stride_o_h, stride_o_t, stride_o_v,
    # Strides for initial state
    stride_initial_state_h, stride_initial_state_k, stride_initial_state_v,
    # Strides for final state
    stride_final_state_h, stride_final_state_k, stride_final_state_v,
    # Other parameters
    scale: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    tl.constexpr,
):
    # Get program indices
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    k_idx = tl.program_id(2)
    v_idx = tl.program_id(3)
    
    # Offsets for K and V blocks
    bk_offsets = tl.arange(0, BK) + k_idx * BK
    bv_offsets = tl.arange(0, BV) + v_idx * BV
    
    # Masks for K and V boundaries
    mask_k = bk_offsets < K
    mask_v = bv_offsets < V
    
    # Initialize hidden state
    if USE_INITIAL_STATE:
        initial_state_offsets = (
            (h_idx * stride_initial_state_h) +
            (bk_offsets[:, None] * stride_initial_state_k) +
            (bv_offsets[None, :] * stride_initial_state_v)
        )
        h = tl.load(initial_state_ptr + initial_state_offsets, mask=mask_k[:, None] & mask_v[None, :], other=0.0)
    else:
        h = tl.zeros((BK, BV), dtype=tl.float32)
    
    # Loop over time steps
    for t in range(T):
        if REVERSE:
            t_load = T - 1 - t
        else:
            t_load = t
        
        # Load k_t [B, H, T, K]
        k_offset = (
            b_idx * stride_k_b +
            h_idx * stride_k_h +
            t_load * stride_k_t
        )
        k_ptr_t = k_ptr + k_offset
        k = tl.load(k_ptr_t + bk_offsets * stride_k_k, mask=mask_k, other=0.0)
        
        # Load v_t [B, H, T, V]
        v_offset = (
            b_idx * stride_v_b +
            h_idx * stride_v_h +
            t_load * stride_v_t
        )
        v_ptr_t = v_ptr + v_offset
        v = tl.load(v_ptr_t + bv_offsets * stride_v_v, mask=mask_v, other=0.0)
        
        # Compute outer product of k and v
        outer = tl.reshape(k, (BK, 1)) * tl.reshape(v, (1, BV))
        
        # Load w [H, K]
        w = tl.load(w_ptr + h_idx * stride_w_h + bk_offsets * stride_w_k, mask=mask_k, other=0.0)
        w = tl.reshape(w, (BK, 1))
        
        # Update hidden state
        h = h * w + outer
        
        # Load q_t [B, H, T, K]
        q_offset = (
            b_idx * stride_q_b +
            h_idx * stride_q_h +
            t_load * stride_q_t
        )
        q_ptr_t = q_ptr + q_offset
        q = tl.load(q_ptr_t + bk_offsets * stride_q_k, mask=mask_k, other=0.0)
        q = q * scale
        
        # Load u [H, V]
        u = tl.load(u_ptr + h_idx * stride_u_h + bv_offsets * stride_u_v, mask=mask_v, other=0.0)
        
        # Compute output
        o_part = tl.sum(q[:, None] * h, axis=0)
        o_part += u * v
        
        # Determine storage index
        t_store = t if REVERSE else t_load
        
        # Store o_t [B, H, T, V]
        o_offset = (
            b_idx * stride_o_b +
            h_idx * stride_o_h +
            t_store * stride_o_t
        )
        o_ptr_t = o_ptr + o_offset
        tl.store(o_ptr_t + bv_offsets * stride_o_v, o_part, mask=mask_v)
    
    # Store final hidden state
    if STORE_FINAL_STATE:
        final_state_offsets = (
            h_idx * stride_final_state_h +
            bk_offsets[:, None] * stride_final_state_k +
            bv_offsets[None, :] * stride_final_state_v
        )
        tl.store(final_state_ptr + final_state_offsets, h, mask=mask_k[:, None] & mask_v[None, :])

class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, u, initial_state, reverse, scale):
        # Save tensors for backward
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.reverse = reverse
        ctx.scale = scale
        
        B, H, T, K = q.shape
        _, _, _, V = v.shape
        
        BK, BV = 32, 32  # Tunable
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        grid = (B, H, NK, NV)
        
        o = torch.empty_like(v, memory_format=torch.contiguous_format)
        final_state = torch.empty((H, K, V), dtype=q.dtype, device=q.device) if initial_state is not None else torch.empty(0, device=q.device)
        
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u,
            o,
            initial_state if initial_state is not None else torch.empty(0, device=q.device),
            final_state,
            B, H, T, K, V,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            w.stride(0), w.stride(1),
            u.stride(0), u.stride(1),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            initial_state.stride(0) if initial_state is not None else 0,
            initial_state.stride(1) if initial_state is not None else 0,
            initial_state.stride(2) if initial_state is not None else 0,
            final_state.stride(0), final_state.stride(1), final_state.stride(2),
            scale,
            BK, BV,
            reverse,
            initial_state is not None,
            initial_state is not None
        )
        return o, final_state if initial_state is not None else o

def fused_recurrent_rwkv6(q, k, v, w, u, initial_state=None, reverse=False, scale=1.0):
    o, final_state = FusedRecurrentRWKV6Function.apply(q, k, v, w, u, initial_state, reverse, scale)
    return o, final_state
