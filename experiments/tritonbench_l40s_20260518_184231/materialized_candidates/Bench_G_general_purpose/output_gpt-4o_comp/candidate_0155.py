import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    q_ptr, k_ptr, v_ptr, h_ptr,
    T, K, V,
    stride_qt, stride_kt, stride_vt, stride_ht,
    stride_qk, stride_vk, stride_hk,
    stride_qv, stride_vv, stride_hv,
    **meta
):
    # Block index
    block_idx = tl.program_id(0)
    
    # Calculate the start of the block in the query, key, and value tensors
    q_start = block_idx * meta['BT']
    k_start = 0
    v_start = 0
    
    # Initialize h
    h = tl.zeros([meta['BT'], meta['BV']], dtype=tl.float32)
    
    # Iterate over time steps to update h
    for t in range(T):
        q = tl.load(q_ptr + (q_start + t) * stride_qt)
        k = tl.load(k_ptr + (k_start + t) * stride_kt)
        v = tl.load(v_ptr + (v_start + t) * stride_vt)
        
        # Compute dot product and update h
        dot_product = tl.dot(q, k)
        h += dot_product * v
    
    # Store the result in h_ptr
    tl.store(h_ptr + q_start * stride_ht, h)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q_ptr, k_ptr, h_ptr, o_ptr,
    T, K, V,
    stride_qt, stride_kt, stride_ht, stride_ot,
    stride_qk, stride_hk, stride_ok,
    stride_qv, stride_hv, stride_ov,
    **meta
):
    # Block index
    block_idx = tl.program_id(0)
    
    # Calculate the start of the block in the query tensor
    q_start = block_idx * meta['BT']
    
    # Initialize output
    o = tl.zeros([meta['BT'], meta['BV']], dtype=tl.float32)
    
    # Iterate over dimensions to compute output
    for t in range(T):
        q = tl.load(q_ptr + (q_start + t) * stride_qt)
        k = tl.load(k_ptr + t * stride_kt)
        h = tl.load(h_ptr + t * stride_ht)
        
        # Compute weighted sum
        dot_product = tl.dot(q, k)
        attention_score = tl.softmax(dot_product)
        o += attention_score * h
    
    # Store the result in o_ptr
    tl.store(o_ptr + q_start * stride_ot, o)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    grad_o_ptr, k_ptr, grad_h_ptr,
    T, K, V,
    stride_got, stride_kt, stride_ght,
    stride_gok, stride_hk,
    **meta
):
    # Block index
    block_idx = tl.program_id(0)
    
    # Calculate the start of the block in the grad_o tensor
    grad_o_start = block_idx * meta['BT']
    
    # Initialize gradient of h
    grad_h = tl.zeros([meta['BT'], meta['BV']], dtype=tl.float32)
    
    # Iterate over dimensions to compute grad_h
    for t in range(T):
        grad_o = tl.load(grad_o_ptr + (grad_o_start + t) * stride_got)
        k = tl.load(k_ptr + t * stride_kt)
        
        # Propagate gradients
        grad_h += tl.dot(grad_o, k)
    
    # Store the result in grad_h_ptr
    tl.store(grad_h_ptr + grad_o_start * stride_ght, grad_h)

@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    grad_o_ptr, q_ptr, k_ptr, v_ptr, grad_q_ptr, grad_k_ptr, grad_v_ptr,
    T, K, V,
    stride_got, stride_qt, stride_kt, stride_vt,
    stride_gqt, stride_gkt, stride_gvt,
    **meta
):
    # Block index
    block_idx = tl.program_id(0)
    
    # Calculate the start of the block in the grad_o tensor
    grad_o_start = block_idx * meta['BT']
    
    # Initialize gradients
    grad_q = tl.zeros([meta['BT'], meta['BK']], dtype=tl.float32)
    grad_k = tl.zeros([meta['BK'], meta['BV']], dtype=tl.float32)
    grad_v = tl.zeros([meta['BV']], dtype=tl.float32)
    
    # Iterate over dimensions to compute gradients
    for t in range(T):
        grad_o = tl.load(grad_o_ptr + (grad_o_start + t) * stride_got)
        q = tl.load(q_ptr + t * stride_qt)
        k = tl.load(k_ptr + t * stride_kt)
        v = tl.load(v_ptr + t * stride_vt)
        
        # Compute gradients
        grad_q += tl.dot(grad_o, k)
        grad_k += tl.dot(q, grad_o)
        grad_v += tl.dot(grad_o, v)
    
    # Store the results in grad_q_ptr, grad_k_ptr, grad_v_ptr
    tl.store(grad_q_ptr + grad_o_start * stride_gqt, grad_q)
    tl.store(grad_k_ptr + grad_o_start * stride_gkt, grad_k)
    tl.store(grad_v_ptr + grad_o_start * stride_gvt, grad_v)

import torch
from torch.autograd import Function

class ChunkLinearAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, T, K, V, BT, BK, BV):
        # Allocate memory for intermediate and output tensors
        h = torch.empty((q.size(0), q.size(1), BV), device=q.device, dtype=q.dtype)
        o = torch.empty_like(q)
        
        # Launch Triton kernels
        grid = (q.size(0) // BT,)
        chunk_linear_attn_fwd_kernel_h[grid](q, k, v, h, T, K, V, q.stride(0), k.stride(0), v.stride(0), h.stride(0))
        chunk_linear_attn_fwd_kernel_o[grid](q, k, h, o, T, K, V, q.stride(0), k.stride(0), h.stride(0), o.stride(0))
        
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, h)
        ctx.T, ctx.K, ctx.V = T, K, V
        ctx.BT, ctx.BK, ctx.BV = BT, BK, BV
        
        return o

    @staticmethod
    def backward(ctx, grad_o):
        q, k, v, h = ctx.saved_tensors
        T, K, V = ctx.T, ctx.K, ctx.V
        BT, BK, BV = ctx.BT, ctx.BK, ctx.BV
        
        # Allocate memory for gradients
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_h = torch.empty_like(h)
        
        # Launch Triton kernels for backward pass
        grid = (q.size(0) // BT,)
        chunk_linear_attn_bwd_kernel_dh[grid](grad_o, k, grad_h, T, K, V, grad_o.stride(0), k.stride(0), grad_h.stride(0))
        chunk_linear_attn_bwd_kernel_dqkv[grid](grad_o, q, k, v, grad_q, grad_k, grad_v, T, K, V, grad_o.stride(0), q.stride(0), k.stride(0), v.stride(0))
        
        return grad_q, grad_k, grad_v, None, None, None, None, None, None
