tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))

class FusedChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        use_init_state = initial_state is not None
        store_final_state = output_final_state and (initial_state is not None)
        initial_state = initial_state if not use_init_state else None

        B, H, T, D_k = q.shape
        DV = v.shape[-1]
        BT = 64
        NK = triton.cdiv(D_k, BT)
        BV = min(triton.next_power_of_2(DV), 64)
        NT = triton.cdiv(T, BT)
        num_stages = 2 if D_k < 512 else 1
        num_warps = 4

        grid = (NK, NT, B*H)
        final_state = torch.empty(B, H, D_k, DV, dtype=q.dtype, device=q.device) if store_final_state else None
        o = torch.empty(B, H, T, DV, dtype=q.dtype, device=q.device)
        fused_chunk_retention_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BT=BT, DK=D_k, DV=DV, BV=BV, BK=BT,
            USE_INITIAL_STATE=use_init_state,
            STORE_FINAL_STATE=store_final_state,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        use_init_state = d_final_state is not None

        B, H, T, D_k = q.shape
        DV = v.shape[-1]
        BT = 64
        NK = triton.cdiv(D_k, BT)
        BV = min(triton.next_power_of_2(DV), 64)
        NT = triton.cdiv(T, BT)
        num_stages = 2 if D_k < 512 else 1
        num_warps = 4

        grid = (NK, NT, B*H)
        dq = torch.empty(B, H, T, D_k, dtype=q.dtype, device=q.device)
        dk = torch.empty(B, H, T, D_k, dtype=q.dtype, device=q.device)
        dv = torch.empty(B, H, T, DV, dtype=q.dtype, device=q.device)

        fused_chunk_retention_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, d_final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            B, H, T, scale,
            BT=BT, DK=D_k, DV=DV, BV=BV, BK=BT,
            USE_INITIAL_STATE=use_init_state,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None

def fused_chunk_retention_func(q, k, v, scale=1.0, initial_state=None, output_final_state=False):
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedChunkRetentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    return o, final_state
