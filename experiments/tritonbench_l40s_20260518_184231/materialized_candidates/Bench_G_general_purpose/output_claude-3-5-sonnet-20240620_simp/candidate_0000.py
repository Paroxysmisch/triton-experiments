import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N, M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Matrix multiplication Q @ K.T
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(N, BLOCK_K)
    
    # Get the block indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Initialize pointers to Q, K, V
    offs_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load Q block
    q = tl.load(Q + offs_q[:, None] * stride_qm + 
                tl.arange(0, BLOCK_K)[None, :] * stride_qk,
                mask=offs_q[:, None] < M)
    
    # Load K block
    k = tl.load(K + offs_k[:, None] * stride_kn + 
                tl.arange(0, BLOCK_K)[None, :] * stride_kk,
                mask=offs_k[:, None] < N)
    
    # Compute attention scores
    qk = tl.dot(q, tl.trans(k))
    qk = qk * (1.0 / tl.sqrt(BLOCK_K))
    
    # Load V block
    v = tl.load(V + offs_v[:, None] * stride_vn + 
                tl.arange(0, BLOCK_K)[None, :] * stride_vk,
                mask=offs_v[:, None] < N)
    
    # Compute output
    out = tl.dot(qk, v)
    
    # Store output
    offs_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(Out + offs_out[:, None] * stride_om + 
             tl.arange(0, BLOCK_K)[None, :] * stride_on,
             out, mask=offs_out[:, None] < M)


@triton.jit
def _bwd_intra_kernel(
    Q, K, V, dO, dQ, dK, dV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N, M,
    BLOCK: tl.constexpr, CBLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK)
    
    # Get block index
    pid_m = pid % num_pid_m
    
    # Initialize pointers
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    mask = offs_m < M
    
    # Load Q, K, V blocks
    q = tl.load(Q + offs_m[:, None] * stride_qm, mask=mask[:, None])
    k = tl.load(K + offs_m[:, None] * stride_kn, mask=mask[:, None])
    v = tl.load(V + offs_m[:, None] * stride_vn, mask=mask[:, None])
    
    # Load dO (gradient of output)
    do = tl.load(dO + offs_m[:, None] * stride_qm, mask=mask[:, None])
    
    # Compute gradients
    dq = tl.dot(do, tl.trans(k)) / tl.sqrt(BLOCK)
    dk = tl.dot(tl.trans(q), do) / tl.sqrt(BLOCK)
    dv = tl.dot(tl.trans(q), do)
    
    # Store gradients
    tl.store(dQ + offs_m[:, None] * stride_qm, dq, mask=mask[:, None])
    tl.store(dK + offs_m[:, None] * stride_kn, dk, mask=mask[:, None])
    tl.store(dV + offs_m[:, None] * stride_vn, dv, mask=mask[:, None])

@triton.jit
def _bwd_inter_kernel(
    Q, K, V, dO, dK, dV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N, M,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK)
    
    # Get block index
    pid_n = pid % num_pid_n
    
    # Initialize pointers
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs_n < N
    
    # Load K, V blocks
    k = tl.load(K + offs_n[:, None] * stride_kn, mask=mask[:, None])
    v = tl.load(V + offs_n[:, None] * stride_vn, mask=mask[:, None])
    
    # Initialize accumulators
    dk_acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    
    # Iterate over M dimension
    for m in range(0, M, BLOCK):
        offs_m = m + tl.arange(0, BLOCK)
        mask_m = offs_m < M
        
        # Load Q and dO blocks
        q = tl.load(Q + offs_m[:, None] * stride_qm, mask=mask_m[:, None])
        do = tl.load(dO + offs_m[:, None] * stride_qm, mask=mask_m[:, None])
        
        # Accumulate gradients
        dk_acc += tl.dot(tl.trans(q), do)
        dv_acc += tl.dot(tl.trans(do), q)
    
    # Store accumulated gradients
    tl.store(dK + offs_n[:, None] * stride_kn, dk_acc, mask=mask[:, None])
    tl.store(dV + offs_n[:, None] * stride_vn, dv_acc, mask=mask[:, None])

import torch
import triton
import triton.language as tl
from typing import Optional

class LightningAttention2NoDecay(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.BLOCK = 64
        self.CBLOCK = 32

    def forward(self, q, k, v):
        batch_size, n_heads, seq_len, d_head = q.shape
        
        # Reshape inputs
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Allocate output
        output = torch.empty_like(q)
        
        # Launch kernel
        grid = (triton.cdiv(seq_len, self.BLOCK), batch_size * n_heads)
        _fwd_kernel[grid](
            q, k, v, output,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            batch_size, n_heads, seq_len, d_head,
            BLOCK_M=self.BLOCK, BLOCK_N=self.BLOCK, BLOCK_K=d_head,
        )
        
        return output

    def backward(self, grad_output, q, k, v):
        batch_size, n_heads, seq_len, d_head = q.shape
        
        # Allocate gradients
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch intra-block kernel
        grid = (triton.cdiv(seq_len, self.BLOCK), batch_size * n_heads)
        _bwd_intra_kernel[grid](
            q, k, v, grad_output, dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, d_head,
            BLOCK=self.BLOCK, CBLOCK=self.CBLOCK,
        )
        
        # Launch inter-block kernel
        grid = (triton.cdiv(seq_len, self.BLOCK), batch_size * n_heads)
        _bwd_inter_kernel[grid](
            q, k, v, grad_output, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, d_head,
            BLOCK=self.BLOCK,
        )
        
        return dq, dk, dv
