import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_on,
    B, H, M, N, K,
    BLOCK: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H
    
    # Initialize pointers to Q, K, V
    q_ptr = Q + bid * stride_qb + hid * stride_qh
    k_ptr = K + bid * stride_kb + hid * stride_kh
    v_ptr = V + bid * stride_vb + hid * stride_vh
    
    # Initialize output pointer
    out_ptr = Out + bid * stride_ob + hid * stride_oh
    
    # Load Q block
    offs_m = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K)
    
    # Mask for bounds checking
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, 
                mask=m_mask[:, None], other=0.0)
                
    # Loop over K,V blocks
    for n in range(0, N, BLOCK):
        k = tl.load(k_ptr + offs_k[:, None] * stride_kk + (n + offs_n)[None, :] * stride_kn,
                    mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptr + (n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                    mask=n_mask[:, None], other=0.0)
        
        # Compute attention scores
        scores = tl.dot(q, k)
        scores = tl.softmax(scores)
        
        # Compute output
        acc += tl.dot(scores, v)
    
    # Store output
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on,
             acc, mask=m_mask[:, None])

@triton.jit
def _bwd_intra_kernel(
    DO, Q, K, V, DQ, DK, DV,
    stride_dob, stride_doh, stride_dom, stride_don,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    B, H, M, N, K,
    CBLOCK: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H
    
    # Initialize pointers
    do_ptr = DO + bid * stride_dob + hid * stride_doh
    q_ptr = Q + bid * stride_qb + hid * stride_qh
    k_ptr = K + bid * stride_kb + hid * stride_kh
    v_ptr = V + bid * stride_vb + hid * stride_vh
    dq_ptr = DQ + bid * stride_dqb + hid * stride_dqh
    dk_ptr = DK + bid * stride_dkb + hid * stride_dkh
    dv_ptr = DV + bid * stride_dvb + hid * stride_dvh
    
    # Load blocks and compute gradients
    offs_m = tl.arange(0, CBLOCK)
    offs_n = tl.arange(0, CBLOCK)
    offs_k = tl.arange(0, K)
    
    # Mask for bounds checking
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Load DO block
    do = tl.load(do_ptr + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_don,
                 mask=m_mask[:, None], other=0.0)
    
    # Compute gradients
    for n in range(0, N, CBLOCK):
        # Load Q, K, V blocks
        q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                   mask=m_mask[:, None], other=0.0)
        k = tl.load(k_ptr + offs_k[:, None] * stride_kk + (n + offs_n)[None, :] * stride_kn,
                   mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptr + (n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                   mask=n_mask[:, None], other=0.0)
        
        # Compute gradients
        dq = tl.dot(do, k.transpose())
        dk = tl.dot(q.transpose(), do)
        dv = tl.dot(do.transpose(), q)
        
        # Accumulate gradients
        tl.atomic_add(dq_ptr + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
                     dq, mask=m_mask[:, None])
        tl.atomic_add(dk_ptr + offs_k[:, None] * stride_dkk + (n + offs_n)[None, :] * stride_dkn,
                     dk, mask=n_mask[None, :])
        tl.atomic_add(dv_ptr + (n + offs_n)[:, None] * stride_dvn + offs_k[None, :] * stride_dvk,
                     dv, mask=n_mask[:, None])

import torch
from torch.autograd import Function
import triton
import triton.language as tl
from typing import Optional

class LightningAttention2NoDecay(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Save tensors for backward
        ctx.save_for_backward(q, k, v)
        
        # Get tensor dimensions
        B, H, M, K = q.shape
        _, _, N, _ = k.shape
        
        # Output tensor
        out = torch.empty_like(q)
        
        # Launch parameters
        BLOCK = 64
        grid = (B * H,)
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, M, N, K,
            BLOCK=BLOCK,
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        
        # Get tensor dimensions
        B, H, M, K = q.shape
        _, _, N, _ = k.shape
        
        # Initialize gradient tensors
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        
        # Launch parameters
        CBLOCK = 32
        grid = (B * H,)
        
        # Launch backward kernels
        _bwd_intra_kernel[grid](
            grad_output, q, k, v, grad_q, grad_k, grad_v,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
            B, H, M, N, K,
            CBLOCK=CBLOCK,
        )
        
        return grad_q, grad_k, grad_v

# Usage example
def lightning_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return LightningAttention2NoDecay.apply(q, k, v)
