import triton
import triton.language as tl

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q_ptr, k_ptr, v_ptr, w_ptr, u_ptr, b_w_ptr, b_u_ptr, b_o_ptr, initial_state_ptr, final_state_ptr,
    T, B, H, D, BK, BV, NK, NV, USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    hid = tl.program_id(axis=2)

    block_start_t = pid * BLOCK_SIZE_T
    block_start_b = bid * BLOCK_SIZE_B
    block_start_h = hid * BLOCK_SIZE_H

    for t in range(block_start_t, min(T, block_start_t + BLOCK_SIZE_T)):
        if REVERSE:
            t = T - 1 - t

        for b in range(block_start_b, min(B, block_start_b + BLOCK_SIZE_B)):
            for h in range(block_start_h, min(H, block_start_h + BLOCK_SIZE_H)):
                # Load initial state if USE_INITIAL_STATE is True
                if USE_INITIAL_STATE and t == 0:
                    state = tl.load(initial_state_ptr + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D))
                else:
                    state = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

                # Load q, k, v, w, u, b_w, b_u
                q = tl.load(q_ptr + t * B * H * D + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D))
                k = tl.load(k_ptr + t * B * H * D + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D))
                v = tl.load(v_ptr + t * B * H * D + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D))
                w = tl.load(w_ptr + t * B * H * D + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D))
                u = tl.load(u_ptr + t * B * H * D + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D))
                b_w = tl.load(b_w_ptr + h * D + tl.arange(0, BLOCK_SIZE_D))
                b_u = tl.load(b_u_ptr + h * D + tl.arange(0, BLOCK_SIZE_D))

                # Compute the recurrent update
                state = state * tl.exp(w) + q * k * b_w + u * b_u
                output = state * v

                # Store the output
                tl.store(b_o_ptr + t * B * H * D + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D), output)

                # Store the final state if STORE_FINAL_STATE is True
                if STORE_FINAL_STATE and t == T - 1:
                    tl.store(final_state_ptr + b * H * D + h * D + tl.arange(0, BLOCK_SIZE_D), state)

import torch
import triton
import triton.language as tl

class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, u, b_w, b_u, initial_state, T, B, H, D, BK, BV, NK, NV, USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE, scale=1.0):
        # Scale q if necessary
        q = q * scale

        # Allocate memory for output and final state
        b_o = torch.empty((T, B, H, D), device=q.device, dtype=q.dtype)
        final_state = torch.empty((B, H, D), device=q.device, dtype=q.dtype) if STORE_FINAL_STATE else None

        # Grid parameters
        grid = (triton.cdiv(T, BK), triton.cdiv(B, BV), triton.cdiv(H, NK))

        # Launch the kernel
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u, b_w, b_u, b_o, initial_state, final_state,
            T, B, H, D, BK, BV, NK, NV, USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE,
            BLOCK_SIZE_T=32, BLOCK_SIZE_B=16, BLOCK_SIZE_H=16, BLOCK_SIZE_D=16
        )

        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, w, u, b_w, b_u, initial_state, b_o, final_state)
        ctx.grid = grid
        ctx.T, ctx.B, ctx.H, ctx.D = T, B, H, D
        ctx.BK, ctx.BV, ctx.NK, ctx.NV = BK, BV, NK, NV
        ctx.USE_INITIAL_STATE, ctx.STORE_FINAL_STATE, ctx.REVERSE = USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE

        return b_o, final_state

def fused_recurrent_rwkv6(q, k, v, w, u, b_w, b_u, initial_state, T, B, H, D, BK, BV, NK, NV, USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE, scale=1.0):
    return FusedRecurrentRWKV6Function.apply(q, k, v, w, u, b_w, b_u, initial_state, T, B, H, D, BK, BV, NK, NV, USE_INITIAL_STATE, STORE_FINAL_STATE, REVERSE, scale)
