import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr,
    T, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_k = pid * BLOCK_SIZE
    offset_v = pid * BLOCK_SIZE
    offset_h = pid * BLOCK_SIZE

    h = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(0, T, BLOCK_SIZE):
        k = tl.load(k_ptr + offset_k + t)
        v = tl.load(v_ptr + offset_v + t)
        h += k * v

    tl.store(h_ptr + offset_h, h)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    T, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_q = pid * BLOCK_SIZE
    offset_k = pid * BLOCK_SIZE
    offset_v = pid * BLOCK_SIZE
    offset_h = pid * BLOCK_SIZE
    offset_o = pid * BLOCK_SIZE

    q = tl.load(q_ptr + offset_q)
    h = tl.load(h_ptr + offset_h)

    o = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(0, T, BLOCK_SIZE):
        k = tl.load(k_ptr + offset_k + t)
        v = tl.load(v_ptr + offset_v + t)
        o += q * (k * v / h)

    tl.store(o_ptr + offset_o, o)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    grad_o_ptr, q_ptr, k_ptr, v_ptr, dh_ptr,
    T, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_q = pid * BLOCK_SIZE
    offset_k = pid * BLOCK_SIZE
    offset_v = pid * BLOCK_SIZE
    offset_grad_o = pid * BLOCK_SIZE
    offset_dh = pid * BLOCK_SIZE

    dh = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(0, T, BLOCK_SIZE):
        grad_o = tl.load(grad_o_ptr + offset_grad_o + t)
        q = tl.load(q_ptr + offset_q + t)
        k = tl.load(k_ptr + offset_k + t)
        v = tl.load(v_ptr + offset_v + t)
        dh += grad_o * q * (k * v)

    tl.store(dh_ptr + offset_dh, dh)

@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    grad_o_ptr, q_ptr, k_ptr, v_ptr, h_ptr, dq_ptr, dk_ptr, dv_ptr,
    T, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_q = pid * BLOCK_SIZE
    offset_k = pid * BLOCK_SIZE
    offset_v = pid * BLOCK_SIZE
    offset_grad_o = pid * BLOCK_SIZE
    offset_h = pid * BLOCK_SIZE
    offset_dq = pid * BLOCK_SIZE
    offset_dk = pid * BLOCK_SIZE
    offset_dv = pid * BLOCK_SIZE

    grad_o = tl.load(grad_o_ptr + offset_grad_o)
    h = tl.load(h_ptr + offset_h)

    dq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dk = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(0, T, BLOCK_SIZE):
        q = tl.load(q_ptr + offset_q + t)
        k = tl.load(k_ptr + offset_k + t)
        v = tl.load(v_ptr + offset_v + t)

        dq += grad_o * (k * v / h)
        dk += grad_o * q * (v / h)
        dv += grad_o * q * (k / h)

    tl.store(dq_ptr + offset_dq, dq)
    tl.store(dk_ptr + offset_dk, dk)
    tl.store(dv_ptr + offset_dv, dv)

import torch
from torch.autograd import Function

class ChunkLinearAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        BLOCK_SIZE = 128  # Define block size
        T = q.shape[0]    # Time steps

        # Allocate output tensors
        h = torch.empty_like(q)
        o = torch.empty_like(q)

        # Launch Triton kernels
        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE']),)
        chunk_linear_attn_fwd_kernel_h[grid](k, v, h, T, BLOCK_SIZE=BLOCK_SIZE)
        chunk_linear_attn_fwd_kernel_o[grid](q, k, v, h, o, T, BLOCK_SIZE=BLOCK_SIZE)

        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, h)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, h = ctx.saved_tensors
        BLOCK_SIZE = 128  # Define block size
        T = q.shape[0]    # Time steps

        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dh = torch.empty_like(h)

        # Launch Triton kernels for backward pass
        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE']),)
        chunk_linear_attn_bwd_kernel_dh[grid](grad_o, q, k, v, dh, T, BLOCK_SIZE=BLOCK_SIZE)
        chunk_linear_attn_bwd_kernel_dqkv[grid](grad_o, q, k, v, h, dq, dk, dv, T, BLOCK_SIZE=BLOCK_SIZE)

        return dq, dk, dv

# Usage in PyTorch
class ChunkLinearAttention(torch.nn.Module):
    def forward(self, q, k, v):
        return ChunkLinearAttentionFunction.apply(q, k, v)
