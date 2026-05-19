import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    # Pointers to tensors
    k_ptr, v_ptr, h_ptr, h0_ptr,
    # Tensor dimensions
    T, K, V,
    # Strides
    stride_k_t, stride_k_k,
    stride_v_t, stride_v_v,
    stride_h_k, stride_h_v,
    # Block sizes
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    # Meta-parameters
    HAS_H0: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_v = tl.program_id(2)
    
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_t = pid_chunk * BT + tl.arange(0, BT)
    
    k_ptrs = k_ptr + offs_t[:, None] * stride_k_t + offs_k[None, :] * stride_k_k
    v_ptrs = v_ptr + offs_t[:, None] * stride_v_t + offs_v[None, :] * stride_v_v
    
    h_acc = tl.zeros((BK, BV), dtype=tl.float32)
    for t in range(BT):
        t_mask = (pid_chunk * BT + t) < T
        k = tl.load(k_ptrs[t], mask=t_mask & (offs_k < K), other=0.0)
        v = tl.load(v_ptrs[t], mask=t_mask & (offs_v < V), other=0.0)
        h_acc += tl.dot(k, v, allow_tf32=False)
    
    if HAS_H0:
        h0_ptrs = h0_ptr + offs_k[:, None] * stride_h_k + offs_v[None, :] * stride_h_v
        h0 = tl.load(h0_ptrs, mask=(offs_k[:, None] < K) & (offs_v[None, :] < V), other=0.0)
        h_acc += h0
    
    h_ptrs = h_ptr + pid_chunk * K * V + offs_k[:, None] * stride_h_k + offs_v[None, :] * stride_h_v
    tl.store(h_ptrs, h_acc.to(h_ptr.dtype.element_ty), mask=(offs_k[:, None] < K) & (offs_v[None, :] < V))

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q_ptr, k_ptr, h_ptr, o_ptr,
    T, K, V,
    stride_q_t, stride_q_k,
    stride_k_t, stride_k_k,
    stride_h_k, stride_h_v,
    stride_o_t, stride_o_v,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    MASK_VAL: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_v = tl.program_id(1)
    
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    
    q_ptrs = q_ptr + offs_t[:, None] * stride_q_t + offs_k[None, :] * stride_q_k
    k_ptrs = k_ptr + offs_t[:, None] * stride_k_t + offs_k[None, :] * stride_k_k
    
    o_acc = tl.zeros((BT, BV), dtype=tl.float32)
    for pid_k in range(0, tl.cdiv(K, BK)):
        k_offs = pid_k * BK + offs_k
        q = tl.load(q_ptrs, mask=(offs_t[:, None] < T) & (k_offs[None, :] < K), other=0.0)
        k = tl.load(k_ptrs, mask=(offs_t[:, None] < T) & (k_offs[None, :] < K), other=0.0)
        
        attn = tl.dot(q, k, allow_tf32=False)
        attn = tl.where(attn != MASK_VAL, attn, 0.0)
        
        h_ptrs = h_ptr + pid_k * BK * V + k_offs[:, None] * stride_h_k + offs_v[None, :] * stride_h_v
        h = tl.load(h_ptrs, mask=(k_offs[:, None] < K) & (offs_v[None, :] < V), other=0.0)
        
        o_acc += tl.dot(attn, h, allow_tf32=False)
    
    o_ptrs = o_ptr + offs_t[:, None] * stride_o_t + offs_v[None, :] * stride_o_v
    tl.store(o_ptrs, o_acc.to(o_ptr.dtype.element_ty), mask=(offs_t[:, None] < T) & (offs_v[None, :] < V))

class ChunkLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, h0=None, mask=None, BT=64, BK=64, BV=64):
        # Allocate output tensor
        T, K, V = q.shape[0], k.shape[1], v.shape[2]
        o = torch.empty_like(v)
        h = torch.zeros((T // BT + 1, K, V), device=q.device, dtype=q.dtype)
        
        # Kernel configurations
        grid_h = (triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV))
        grid_o = (triton.cdiv(T, BT), triton.cdiv(V, BV))
        
        # Launch h kernel
        chunk_linear_attn_fwd_kernel_h[grid_h](
            k, v, h, h0,
            T, K, V,
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            h.stride(1), h.stride(2),
            BT, BK, BV,
            h0 is not None,
        )
        
        # Launch o kernel
        MASK_VAL = -float('inf') if mask is not None else 0.0
        chunk_linear_attn_fwd_kernel_o[grid_o](
            q, k, h, o,
            T, K, V,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            h.stride(1), h.stride(2),
            o.stride(0), o.stride(1),
            BT, BK, BV,
            MASK_VAL,
        )
        
        ctx.save_for_backward(q, k, v, h, mask)
        ctx.BT, ctx.BK, ctx.BV = BT, BK, BV
        return o

    @staticmethod
    def backward(ctx, do):
        # Backward implementation with similar structure
        # (Omitted for brevity but follows similar kernel structure)
        raise NotImplementedError("Backward pass not implemented")

def chunk_linear_attention(q, k, v, h0=None, mask=None, BT=64, BK=64, BV=64):
    return ChunkLinearAttentionFunction.apply(q, k, v, h0, mask, BT, BK, BV)
