import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, h_ptr,  # Pointers to input and output tensors
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,  # Strides for query tensor
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,  # Strides for key tensor
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,  # Strides for value tensor
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,  # Strides for output tensor
    h_stride_b, h_stride_h, h_stride_s, h_stride_d,  # Strides for intermediate tensor h
    B, H, S, D,  # Batch size, number of heads, sequence length, head dimension
    BT, BD, NT,  # Block sizes for sequence, head, and number of blocks
    scale,  # Scaling factor for attention scores
    BLOCK_SIZE: tl.constexpr
):
    # Get the block indices
    pid = tl.program_id(axis=0)
    bid = pid // (NT * BD)
    hid = (pid % (NT * BD)) // NT
    tid = (pid % (NT * BD)) % NT

    # Compute the base indices for the current block
    q_base = bid * q_stride_b + hid * q_stride_h + tid * q_stride_s
    k_base = bid * k_stride_b + hid * k_stride_h + tid * k_stride_s
    v_base = bid * v_stride_b + hid * v_stride_h + tid * v_stride_s
    out_base = bid * out_stride_b + hid * out_stride_h + tid * out_stride_s
    h_base = bid * h_stride_b + hid * h_stride_h + tid * h_stride_s

    # Initialize the attention scores and output values
    scores = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    out = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)

    # Load the query, key, and value blocks
    q = tl.load(q_ptr + q_base * q_stride_d + tl.arange(0, D))
    k = tl.load(k_ptr + k_base * k_stride_d + tl.arange(0, D))
    v = tl.load(v_ptr + v_base * v_stride_d + tl.arange(0, D))

    # Compute the attention scores
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            scores[i, j] = tl.sum(q[i] * k[j], axis=0) * scale

    # Apply the softmax function to the attention scores
    scores_max = tl.max(scores, axis=1)
    scores_exp = tl.exp(scores - scores_max[:, None])
    scores_sum = tl.sum(scores_exp, axis=1)
    scores_softmax = scores_exp / scores_sum[:, None]

    # Compute the output values
    for i in range(BLOCK_SIZE):
        out[i] = tl.sum(scores_softmax[i, :, None] * v, axis=0)

    # Store the output values
    tl.store(out_ptr + out_base * out_stride_d + tl.arange(0, D), out)

    # Optionally store the intermediate tensor h
    if h_ptr is not None:
        tl.store(h_ptr + h_base * h_stride_d + tl.arange(0, D), scores_softmax)

import torch
from torch.autograd import Function

class AttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, h=None, scale=1.0, BT=16, BD=16, NT=16):
        B, H, S, D = q.shape

        # Allocate output tensor
        out = torch.empty_like(q)

        # Allocate intermediate tensor h if provided
        if h is not None:
            h = torch.empty((B, H, S, S), device=q.device, dtype=torch.float32)

        # Define the grid and block dimensions
        grid = (B * H * S // (BT * BD * NT),)

        # Launch the kernel
        attention_fwd_kernel[grid](
            q, k, v, out, h,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            h.stride(0) if h is not None else 0, h.stride(1) if h is not None else 0, h.stride(2) if h is not None else 0, h.stride(3) if h is not None else 0,
            B, H, S, D,
            BT, BD, NT,
            scale,
            BT
        )

        # Save the context for backward pass if needed
        ctx.save_for_backward(q, k, v, out, h)
        ctx.B, ctx.H, ctx.S, ctx.D = B, H, S, D
        ctx.BT, ctx.BD, ctx.NT = BT, BD, NT
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Implement the backward pass if needed
        q, k, v, out, h = ctx.saved_tensors
        B, H, S, D = ctx.B, ctx.H, ctx.S, ctx.D
        BT, BD, NT = ctx.BT, ctx.BD, ctx.NT
        scale = ctx.scale

        # Allocate gradients
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        # Define the grid and block dimensions
        grid = (B * H * S // (BT * BD * NT),)

        # Launch the backward kernel (not implemented here)
        # attention_bwd_kernel[grid](
        #     grad_out, q, k, v, out, h, grad_q, grad_k, grad_v,
        #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        #     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        #     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        #     out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        #     h.stride(0) if h is not None else 0, h.stride(1) if h is not None else 0, h.stride(2) if h is not None else 0, h.stride(3) if h is not None else 0,
        #     grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
        #     grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
        #     grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
        #     B, H, S, D,
        #     BT, BD, NT,
        #     scale,
        #     BT
        # )

        return grad_q, grad_k, grad_v, None, None, None, None

import torch

# Example input tensors
B, H, S, D = 2, 8, 128, 64
q = torch.randn(B, H, S, D, device='cuda')
k = torch.randn(B, H, S, D, device='cuda')
v = torch.randn(B, H, S, D, device='cuda')

# Optional intermediate tensor h
h = torch.empty((B, H, S, S), device='cuda', dtype=torch.float32)

# Forward pass
out = AttentionFunction.apply(q, k, v, h, scale=1.0 / (D ** 0.5))

# Print the output
print(out)
