import triton
import triton.language as tl

@triton.jit
def fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, 
    q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1, out_stride0, out_stride1,
    N, H, D, 
    beta, scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    q_offsets = tl.arange(0, H)[:, None] * q_stride0 + offsets[None, :] * q_stride1
    k_offsets = tl.arange(0, H)[:, None] * k_stride0 + offsets[None, :] * k_stride1
    v_offsets = tl.arange(0, H)[:, None] * v_stride0 + offsets[None, :] * v_stride1
    out_offsets = tl.arange(0, H)[:, None] * out_stride0 + offsets[None, :] * out_stride1

    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=mask, other=0.0)

    # Compute the recurrent operation
    out = tl.zeros((H, BLOCK_SIZE), dtype=tl.float32)
    for i in range(D):
        q_i = q * beta
        k_i = k * beta
        v_i = v * beta
        out += q_i * k_i * v_i * scale

    tl.store(out_ptr + out_offsets, out, mask=mask)

@triton.jit
def fused_recurrent_bwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, grad_out_ptr, grad_q_ptr, grad_k_ptr, grad_v_ptr,
    q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1, out_stride0, out_stride1, grad_q_stride0, grad_q_stride1, grad_k_stride0, grad_k_stride1, grad_v_stride0, grad_v_stride1,
    N, H, D, 
    beta, scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    q_offsets = tl.arange(0, H)[:, None] * q_stride0 + offsets[None, :] * q_stride1
    k_offsets = tl.arange(0, H)[:, None] * k_stride0 + offsets[None, :] * k_stride1
    v_offsets = tl.arange(0, H)[:, None] * v_stride0 + offsets[None, :] * v_stride1
    out_offsets = tl.arange(0, H)[:, None] * out_stride0 + offsets[None, :] * out_stride1
    grad_q_offsets = tl.arange(0, H)[:, None] * grad_q_stride0 + offsets[None, :] * grad_q_stride1
    grad_k_offsets = tl.arange(0, H)[:, None] * grad_k_stride0 + offsets[None, :] * grad_k_stride1
    grad_v_offsets = tl.arange(0, H)[:, None] * grad_v_stride0 + offsets[None, :] * grad_v_stride1

    q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + out_offsets, mask=mask, other=0.0)

    grad_q = tl.zeros((H, BLOCK_SIZE), dtype=tl.float32)
    grad_k = tl.zeros((H, BLOCK_SIZE), dtype=tl.float32)
    grad_v = tl.zeros((H, BLOCK_SIZE), dtype=tl.float32)

    for i in range(D):
        q_i = q * beta
        k_i = k * beta
        v_i = v * beta
        grad_out_i = grad_out * scale
        grad_q += grad_out_i * k_i * v_i * beta
        grad_k += grad_out_i * q_i * v_i * beta
        grad_v += grad_out_i * q_i * k_i * beta

    tl.store(grad_q_ptr + grad_q_offsets, grad_q, mask=mask)
    tl.store(grad_k_ptr + grad_k_offsets, grad_k, mask=mask)
    tl.store(grad_v_ptr + grad_v_offsets, grad_v, mask=mask)

import torch
from torch.autograd import Function

class FusedRecurrentFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, scale, initial_state=None):
        N, H, D = q.shape
        out = torch.empty_like(q)

        BLOCK_SIZE = 128
        grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

        fused_recurrent_fwd_kernel[grid, BLOCK_SIZE](
            q, k, v, out,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1),
            N, H, D, beta, scale
        )

        ctx.save_for_backward(q, k, v, out, beta, scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, beta, scale = ctx.saved_tensors
        N, H, D = q.shape

        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        BLOCK_SIZE = 128
        grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

        fused_recurrent_bwd_kernel[grid, BLOCK_SIZE](
            q, k, v, out, grad_out, grad_q, grad_k, grad_v,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0), out.stride(1), grad_q.stride(0), grad_q.stride(1), grad_k.stride(0), grad_k.stride(1), grad_v.stride(0), grad_v.stride(1),
            N, H, D, beta, scale
        )

        return grad_q, grad_k, grad_v, None, None, None

def fused_recurrent_delta_rule(q, k, v, beta=1.0, scale=1.0, initial_state=None):
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Input tensors q, k, and v must have the same shape.")
    if len(q.shape) != 3:
        raise ValueError("Input tensors q, k, and v must be 3-dimensional.")

    return FusedRecurrentFunction.apply(q, k, v, beta, scale, initial_state)

import torch

# Example input tensors
q = torch.randn(10, 8, 64, requires_grad=True, device='cuda')
k = torch.randn(10, 8, 64, requires_grad=True, device='cuda')
v = torch.randn(10, 8, 64, requires_grad=True, device='cuda')

# Parameters
beta = 0.5
scale = 0.1

# Apply the fused recurrent operation
out = fused_recurrent_delta_rule(q, k, v, beta, scale)

# Compute loss and backpropagate
loss = out.sum()
loss.backward()

print("Output:", out)
print("Gradient w.r.t. q:", q.grad)
print("Gradient w.r.t. k:", k.grad)
print("Gradient w.r.t. v:", v.grad)
