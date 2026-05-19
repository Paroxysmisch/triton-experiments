import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(Q, K, V, O, Z, M, N, L, scale, stride_q, stride_k, stride_v, stride_o, stride_z):
    # Define the block size
    BLOCK_SIZE = 128

    # Define program IDs for parallel execution
    pid = tl.program_id(0)

    # Compute the offsets for Q, K, V, O, and Z
    offs_q = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, N)
    offs_v = tl.arange(0, L)

    # Load Q, K, V
    q = tl.load(Q + offs_q[:, None] * stride_q, mask=offs_q[:, None] < M)
    k = tl.load(K + offs_k[None, :] * stride_k, mask=offs_k[None, :] < N)
    v = tl.load(V + offs_v[None, :] * stride_v, mask=offs_v[None, :] < L)

    # Compute attention scores
    attn_scores = tl.dot(q, k) * scale

    # Compute softmax normalization
    max_score = tl.max(attn_scores, axis=1)
    attn_scores = attn_scores - max_score[:, None]
    exp_scores = tl.exp(attn_scores)
    z = tl.sum(exp_scores, axis=1)

    # Store normalizer
    tl.store(Z + offs_q, z, mask=offs_q < M)

    # Compute weighted sum
    o = tl.dot(exp_scores, v)

    # Store output
    tl.store(O + offs_q[:, None] * stride_o, o, mask=offs_q[:, None] < M)

@triton.jit
def parallel_rebased_bwd_kernel(Q, K, V, dO, dQ, dK, dV, Z, M, N, L, scale, stride_q, stride_k, stride_v, stride_do, stride_dq, stride_dk, stride_dv):
    # Define the block size
    BLOCK_SIZE = 128

    # Define program IDs for parallel execution
    pid = tl.program_id(0)

    # Compute the offsets for Q, K, V, dO, dQ, dK, dV
    offs_q = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, N)
    offs_v = tl.arange(0, L)

    # Load Q, K, V, dO, Z
    q = tl.load(Q + offs_q[:, None] * stride_q, mask=offs_q[:, None] < M)
    k = tl.load(K + offs_k[None, :] * stride_k, mask=offs_k[None, :] < N)
    v = tl.load(V + offs_v[None, :] * stride_v, mask=offs_v[None, :] < L)
    do = tl.load(dO + offs_q[:, None] * stride_do, mask=offs_q[:, None] < M)
    z = tl.load(Z + offs_q, mask=offs_q < M)

    # Compute gradients
    dq = tl.zeros_like(q)
    dk = tl.zeros_like(k)
    dv = tl.zeros_like(v)

    # Backprop through weighted sum
    dv += tl.dot(do, v)

    # Backprop through softmax
    grad_exp_scores = tl.dot(do, k) * scale / z[:, None]
    grad_attn_scores = grad_exp_scores - tl.sum(grad_exp_scores, axis=1)[:, None]
    
    # Backprop through dot product
    dq += tl.dot(grad_attn_scores, k)
    dk += tl.dot(q, grad_attn_scores)

    # Store gradients
    tl.store(dQ + offs_q[:, None] * stride_dq, dq, mask=offs_q[:, None] < M)
    tl.store(dK + offs_k[None, :] * stride_dk, dk, mask=offs_k[None, :] < N)
    tl.store(dV + offs_v[None, :] * stride_dv, dv, mask=offs_v[None, :] < L)

import torch

class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale):
        M, N, L = Q.shape[0], K.shape[1], V.shape[1]
        O = torch.empty((M, L), device=Q.device, dtype=Q.dtype)
        Z = torch.empty((M,), device=Q.device, dtype=Q.dtype)
        
        # Launch the forward kernel
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
        parallel_rebased_fwd_kernel[grid](
            Q, K, V, O, Z, M, N, L, scale,
            Q.stride(0), K.stride(0), V.stride(0),
            O.stride(0), Z.stride(0)
        )
        
        # Save context for backward
        ctx.save_for_backward(Q, K, V, Z)
        ctx.scale = scale
        
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, Z = ctx.saved_tensors
        M, N, L = Q.shape[0], K.shape[1], V.shape[1]
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        
        # Launch the backward kernel
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
        parallel_rebased_bwd_kernel[grid](
            Q, K, V, dO, dQ, dK, dV, Z, M, N, L, ctx.scale,
            Q.stride(0), K.stride(0), V.stride(0),
            dO.stride(0), dQ.stride(0), dK.stride(0), dV.stride(0)
        )
        
        return dQ, dK, dV, None

def parallel_rebased(Q, K, V, scale=1.0):
    return ParallelBasedFunction.apply(Q, K, V, scale)
