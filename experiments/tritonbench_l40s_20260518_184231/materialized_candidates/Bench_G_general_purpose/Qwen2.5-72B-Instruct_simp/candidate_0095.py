import triton
import triton.language as tl

@triton.jit
def fused_recurrent_retention_forward_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, initial_state_ptr, final_state_ptr,
    q_batch_stride, q_seq_stride, q_head_stride, q_feat_stride,
    k_batch_stride, k_seq_stride, k_head_stride, k_feat_stride,
    v_batch_stride, v_seq_stride, v_head_stride, v_feat_stride,
    o_batch_stride, o_seq_stride, o_head_stride, o_feat_stride,
    initial_state_batch_stride, initial_state_head_stride, initial_state_feat_stride,
    final_state_batch_stride, final_state_head_stride, final_state_feat_stride,
    B, T, H, D, BK, BV, USE_INITIAL, STORE_FINAL, BLOCK_SIZE_T: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    start = tl.program_id(2) * BLOCK_SIZE_T

    q_offset = b * q_batch_stride + h * q_head_stride
    k_offset = b * k_batch_stride + h * k_head_stride
    v_offset = b * v_seq_stride + h * v_head_stride
    o_offset = b * o_batch_stride + h * o_head_stride

    if USE_INITIAL:
        initial_state = tl.load(initial_state_ptr + b * initial_state_batch_stride + h * initial_state_head_stride)

    h = tl.zeros((BK, BV), dtype=tl.float32)

    for t in range(start, start + BLOCK_SIZE_T):
        if t < T:
            q = tl.load(q_ptr + q_offset + t * q_seq_stride, mask=t < T, other=0.0)
            k = tl.load(k_ptr + k_offset + t * k_seq_stride, mask=t < T, other=0.0)
            v = tl.load(v_ptr + v_offset + t * v_seq_stride, mask=t < T, other=0.0)

            q = q * 1.0 / (D ** 0.5)  # Scale the query

            h = h + tl.dot(q, k)  # Update the key-value product
            o = tl.dot(h, v)  # Compute the output

            tl.store(o_ptr + o_offset + t * o_seq_stride, o, mask=t < T)

            if STORE_FINAL and t == T - 1:
                tl.store(final_state_ptr + b * final_state_batch_stride + h * final_state_head_stride, h)

@triton.jit
def fused_recurrent_retention_backward_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
    q_batch_stride, q_seq_stride, q_head_stride, q_feat_stride,
    k_batch_stride, k_seq_stride, k_head_stride, k_feat_stride,
    v_batch_stride, v_seq_stride, v_head_stride, v_feat_stride,
    do_batch_stride, do_seq_stride, do_head_stride, do_feat_stride,
    dq_batch_stride, dq_seq_stride, dq_head_stride, dq_feat_stride,
    dk_batch_stride, dk_seq_stride, dk_head_stride, dk_feat_stride,
    dv_batch_stride, dv_seq_stride, dv_head_stride, dv_feat_stride,
    B, T, H, D, BK, BV, BLOCK_SIZE_T: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    start = tl.program_id(2) * BLOCK_SIZE_T

    q_offset = b * q_batch_stride + h * q_head_stride
    k_offset = b * k_batch_stride + h * k_head_stride
    v_offset = b * v_seq_stride + h * v_head_stride
    do_offset = b * do_batch_stride + h * do_head_stride
    dq_offset = b * dq_batch_stride + h * dq_head_stride
    dk_offset = b * dk_batch_stride + h * dk_head_stride
    dv_offset = b * dv_batch_stride + h * dv_head_stride

    h = tl.zeros((BK, BV), dtype=tl.float32)

    for t in range(T - 1, start - 1, -1):
        if t >= start:
            q = tl.load(q_ptr + q_offset + t * q_seq_stride, mask=t >= start, other=0.0)
            k = tl.load(k_ptr + k_offset + t * k_seq_stride, mask=t >= start, other=0.0)
            v = tl.load(v_ptr + v_offset + t * v_seq_stride, mask=t >= start, other=0.0)
            do = tl.load(do_ptr + do_offset + t * do_seq_stride, mask=t >= start, other=0.0)

            q = q * 1.0 / (D ** 0.5)  # Scale the query

            dv = tl.dot(h, do)
            h = h + tl.dot(q, k)
            dk = tl.dot(q, do)
            dq = tl.dot(do, v) * (1.0 / (D ** 0.5))

            tl.store(dq_ptr + dq_offset + t * dq_seq_stride, dq, mask=t >= start)
            tl.store(dk_ptr + dk_offset + t * dk_seq_stride, dk, mask=t >= start)
            tl.store(dv_ptr + dv_offset + t * dv_seq_stride, dv, mask=t >= start)

import torch
from torch.autograd import Function

class FusedRecurrentRetentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, store_final=False):
        B, T, H, D = q.shape
        BK, BV = k.shape[-1], v.shape[-1]

        o = torch.empty_like(q)
        final_state = torch.empty((B, H, BK, BV), device=q.device) if store_final else None

        USE_INITIAL = initial_state is not None
        STORE_FINAL = store_final

        grid = (B, H, (T + 128 - 1) // 128)
        fused_recurrent_retention_forward_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            initial_state.stride(0) if USE_INITIAL else 0, initial_state.stride(1) if USE_INITIAL else 0, initial_state.stride(2) if USE_INITIAL else 0,
            final_state.stride(0) if STORE_FINAL else 0, final_state.stride(1) if STORE_FINAL else 0, final_state.stride(2) if STORE_FINAL else 0,
            B, T, H, D, BK, BV, USE_INITIAL, STORE_FINAL, 128
        )

        ctx.save_for_backward(q, k, v, o)
        ctx.B, ctx.T, ctx.H, ctx.D, ctx.BK, ctx.BV = B, T, H, D, BK, BV
        ctx.STORE_FINAL = STORE_FINAL

        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, o = ctx.saved_tensors
        B, T, H, D = ctx.B, ctx.T, ctx.H, ctx.D
        BK, BV = ctx.BK, ctx.BV

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        grid = (B, H, (T + 128 - 1) // 128)
        fused_recurrent_retention_backward_kernel[grid](
            q, k, v, do, dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            B, T, H, D, BK, BV, 128
        )

        return dq, dk, dv, None, None
