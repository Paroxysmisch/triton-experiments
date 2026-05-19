import torch
import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    # Tensor dimensions and strides
    B, H, T, D,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_h_b, stride_h_h, stride_h_t, stride_h_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    scale,
    BT: tl.constexpr,  # Block size for sequence length
    BD: tl.constexpr,  # Block size for head dimension
    STORE_H: tl.constexpr,
    IFCOND: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs_t = pid_t * BT + tl.arange(0, BT)
    mask_t = offs_t < T

    # Load query block
    q_offs = pid_b * stride_q_b + pid_h * stride_q_h + offs_t[:, None] * stride_q_t + tl.arange(0, BD)[None, :] * stride_q_d
    q = tl.load(q_ptr + q_offs, mask=mask_t[:, None] & (tl.arange(0, BD)[None, :] < D), other=0.0)

    m_i = tl.zeros([BT], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BT], dtype=tl.float32)
    acc = tl.zeros([BT, BD], dtype=tl.float32)

    num_blocks = tl.cdiv(T, BT)
    for i in range(num_blocks):
        offs_i = i * BT + tl.arange(0, BT)
        mask_i = offs_i < T

        # Load key block
        k_offs = pid_b * stride_k_b + pid_h * stride_k_h + offs_i[None, :] * stride_k_t + tl.arange(0, BD)[:, None] * stride_k_d
        k = tl.load(k_ptr + k_offs, mask=mask_i[None, :] & (tl.arange(0, BD)[:, None] < D), other=0.0)

        # Compute scores for current block
        s = tl.dot(q, k) * scale

        # Update max and sum for online softmax
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        alpha = tl.exp(m_i - m_curr)
        p = tl.exp(s - m_curr[:, None])

        l_curr = alpha * l_i + tl.sum(p, axis=1)

        # Load value block
        v_offs = pid_b * stride_v_b + pid_h * stride_v_h + offs_i[:, None] * stride_v_t + tl.arange(0, BD)[None, :] * stride_v_d
        v = tl.load(v_ptr + v_offs, mask=mask_i[:, None] & (tl.arange(0, BD)[None, :] < D), other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # Update m_i and l_i for next iteration
        m_i = m_curr
        l_i = l_curr

    # Normalize accumulator
    acc = acc / l_i[:, None]

    # Store output
    o_offs = pid_b * stride_o_b + pid_h * stride_o_h + offs_t[:, None] * stride_o_t + tl.arange(0, BD)[None, :] * stride_o_d
    tl.store(o_ptr + o_offs, acc, mask=mask_t[:, None] & (tl.arange(0, BD)[None, :] < D))

    # Store h if needed
    if STORE_H:
        h_offs = pid_b * stride_h_b + pid_h * stride_h_h + offs_t[:, None] * stride_h_t + tl.arange(0, 2)[None, :] * stride_h_d
        h = tl.stack([m_i, l_i], axis=1)
        tl.store(h_ptr + h_offs, h, mask=mask_t[:, None] & (tl.arange(0, 2)[None, :] < 2))

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, store_h=False, if_cond=False):
        B, H, T, D = q.shape
        scale = D ** -0.5

        BT = 64  # Tune based on hardware
        BD = min(128, D)  # Ensure BD <= D

        o = torch.empty_like(q)
        h = torch.empty((B, H, T, 2), device=q.device, dtype=torch.float32) if store_h else None

        grid = (B, H, (T + BT - 1) // BT)
        num_warps = 4 if BD <= 64 else 8
        num_stages = 3 if BD <= 64 else 2

        attention_fwd_kernel[grid](
            q, k, v, h if store_h else None, o,
            B, H, T, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            h.stride(0) if store_h else 0, h.stride(1) if store_h else 0,
            h.stride(2) if store_h else 0, h.stride(3) if store_h else 0,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            scale,
            BT=BT, BD=BD,
            STORE_H=store_h,
            IFCOND=if_cond,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, h, o)
        ctx.scale = scale
        ctx.store_h = store_h
        return o

    @staticmethod
    def backward(ctx, do):
        # Backward pass not implemented for this example
        raise NotImplementedError("Backward pass is not implemented.")

def attention(q, k, v, store_h=False, if_cond=False):
    return AttentionFunction.apply(q, k, v, store_h, if_cond)
