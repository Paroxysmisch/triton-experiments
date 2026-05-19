import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(k_ptr, v_ptr, h_ptr, state_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    k = tl.load(k_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    
    h = k * v  # Example computation, replace with the actual operation

    if state_ptr is not None:
        state = tl.load(state_ptr + offs, mask=mask)
        h += state  # Update with initial state if provided

    tl.store(h_ptr + offs, h, mask=mask)

@triton.jit
def chunk_retention_fwd_kernel_o(q_ptr, k_ptr, v_ptr, h_ptr, o_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    q = tl.load(q_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    h = tl.load(h_ptr + offs, mask=mask)

    o = q * k * v + h  # Example computation, replace with the actual operation

    tl.store(o_ptr + offs, o, mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dh(dout_ptr, k_ptr, v_ptr, dh_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    dout = tl.load(dout_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)

    dh = dout * k * v  # Example gradient computation

    tl.store(dh_ptr + offs, dh, mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(dout_ptr, q_ptr, k_ptr, v_ptr, dq_ptr, dk_ptr, dv_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    dout = tl.load(dout_ptr + offs, mask=mask)
    q = tl.load(q_ptr + offs, mask=mask)
    k = tl.load(k_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)

    dq = dout * k * v  # Example gradient computation
    dk = dout * q * v
    dv = dout * q * k

    tl.store(dq_ptr + offs, dq, mask=mask)
    tl.store(dk_ptr + offs, dk, mask=mask)
    tl.store(dv_ptr + offs, dv, mask=mask)

import torch
from torch.autograd import Function

class ChunkRetentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, state=None):
        N, M = q.shape

        h = torch.empty_like(k)
        o = torch.empty_like(q)

        # Launch Triton kernels
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        chunk_retention_fwd_kernel_h[grid](k, v, h, state, N, M, BLOCK_SIZE=1024)
        chunk_retention_fwd_kernel_o[grid](q, k, v, h, o, N, M, BLOCK_SIZE=1024)

        ctx.save_for_backward(q, k, v, h)

        return o

    @staticmethod
    def backward(ctx, dout):
        q, k, v, h = ctx.saved_tensors
        N, M = dout.shape

        dh = torch.empty_like(h)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Launch Triton kernels for backward pass
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        chunk_retention_bwd_kernel_dh[grid](dout, k, v, dh, N, M, BLOCK_SIZE=1024)
        chunk_retention_bwd_kernel_dqkv[grid](dout, q, k, v, dq, dk, dv, N, M, BLOCK_SIZE=1024)

        return dq, dk, dv, None

def chunk_retention(q, k, v, state=None):
    return ChunkRetentionFunction.apply(q, k, v, state)
