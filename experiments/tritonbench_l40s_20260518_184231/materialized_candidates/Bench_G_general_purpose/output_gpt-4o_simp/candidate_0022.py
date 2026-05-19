import triton
import triton.language as tl
import torch

# Constants for block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 128

@triton.jit
def _fwd_kernel(Q, K, V, Out, L, M, sm_scale, stride_qm, stride_qk, stride_km, stride_kn, stride_vm, stride_vn, stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Compute block indices
    pid = tl.program_id(0)
    # Compute block start indices
    start_m = pid * BLOCK_M
    # Load Q, K, V tiles
    q = tl.load(Q + start_m * stride_qm, mask=start_m + tl.arange(0, BLOCK_M) < Q.shape[0])
    k = tl.load(K + tl.arange(0, BLOCK_N) * stride_kn, mask=tl.arange(0, BLOCK_N) < K.shape[1])
    v = tl.load(V + tl.arange(0, BLOCK_N) * stride_vn, mask=tl.arange(0, BLOCK_N) < V.shape[1])
    # Compute QK^T
    qk = tl.dot(q, k)
    # Scale by sm_scale
    qk_scaled = qk * sm_scale
    # Apply softmax
    qk_softmax = tl.softmax(qk_scaled, axis=1)
    # Compute weighted sum
    out = tl.dot(qk_softmax, v)
    # Store result
    tl.store(Out + start_m * stride_om, out, mask=start_m + tl.arange(0, BLOCK_M) < Out.shape[0])

@triton.jit
def _bwd_kernel(dOut, Q, K, V, dQ, dK, dV, L, M, sm_scale, stride_qm, stride_qk, stride_km, stride_kn, stride_vm, stride_vn, stride_dom, stride_don, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    # Compute block indices
    pid = tl.program_id(0)
    # Compute block start indices
    start_m = pid * BLOCK_M
    # Load gradients and inputs
    dout = tl.load(dOut + start_m * stride_dom, mask=start_m + tl.arange(0, BLOCK_M) < dOut.shape[0])
    q = tl.load(Q + start_m * stride_qm, mask=start_m + tl.arange(0, BLOCK_M) < Q.shape[0])
    k = tl.load(K + tl.arange(0, BLOCK_N) * stride_kn, mask=tl.arange(0, BLOCK_N) < K.shape[1])
    v = tl.load(V + tl.arange(0, BLOCK_N) * stride_vn, mask=tl.arange(0, BLOCK_N) < V.shape[1])
    # Compute gradients
    dq = tl.dot(dout, v.T)
    dk = tl.dot(q.T, dout)
    dv = tl.dot(qk_softmax.T, dout)
    # Store gradients
    tl.store(dQ + start_m * stride_qm, dq, mask=start_m + tl.arange(0, BLOCK_M) < dQ.shape[0])
    tl.store(dK + tl.arange(0, BLOCK_N) * stride_kn, dk, mask=tl.arange(0, BLOCK_N) < dK.shape[1])
    tl.store(dV + tl.arange(0, BLOCK_N) * stride_vn, dv, mask=tl.arange(0, BLOCK_N) < dV.shape[1])

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale):
        # Allocate output tensor
        Out = torch.empty((Q.shape[0], V.shape[1]), device=Q.device, dtype=Q.dtype)
        # Launch Triton kernel
        grid = (triton.cdiv(Q.shape[0], BLOCK_M),)
        _fwd_kernel[grid](Q, K, V, Out, None, None, sm_scale, Q.stride(0), Q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1), Out.stride(0), Out.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL)
        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, Out)
        ctx.sm_scale = sm_scale
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q, K, V, Out = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        # Allocate gradient tensors
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        # Launch Triton kernel for backward pass
        grid = (triton.cdiv(Q.shape[0], BLOCK_M),)
        _bwd_kernel[grid](dOut, Q, K, V, dQ, dK, dV, None, None, sm_scale, Q.stride(0), Q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1), dOut.stride(0), dOut.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL)
        return dQ, dK, dV, None

# Usage
def attention(Q, K, V, sm_scale):
    return Attention.apply(Q, K, V, sm_scale)
