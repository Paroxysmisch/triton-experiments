import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, L, m,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    sm_scale,
    causal: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    num_n_blocks = triton.cdiv(N_CTX, BLOCK_N)
    
    # Block ID
    bid_z = pid // (H * num_m_blocks)
    bid_h = (pid % (H * num_m_blocks)) // num_m_blocks
    bid_m = (pid % (H * num_m_blocks)) % num_m_blocks
    
    # Initialize pointers to Q, K, V
    q_start = Q + bid_z * stride_qz + bid_h * stride_qh + bid_m * BLOCK_M * stride_qm
    k_start = K + bid_z * stride_kz + bid_h * stride_kh
    v_start = V + bid_z * stride_vz + bid_h * stride_vh
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    max_val = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    sum_val = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(q_start + tl.arange(0, BLOCK_M)[:, None] * stride_qm +
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk)
    
    # Loop over K,V blocks
    for block_n in range(0, num_n_blocks):
        k_ptr = k_start + block_n * BLOCK_N * stride_kn
        v_ptr = v_start + block_n * BLOCK_N * stride_vn
        
        # Load K,V blocks
        k = tl.load(k_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_kn +
                   tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk)
        v = tl.load(v_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_vn +
                   tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * sm_scale
        
        # Apply causal mask if needed
        if causal:
            row_idx = bid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
            col_idx = block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            mask = col_idx <= row_idx
            scores = tl.where(mask, scores, float("-inf"))
        
        # Update max values and compute exponentials
        block_max = tl.max(scores, 1)
        max_val = tl.maximum(max_val, block_max)
        exp_scores = tl.exp(scores - max_val[:, None])
        sum_val += tl.sum(exp_scores, 1)
        
        # Update accumulator
        acc += tl.dot(exp_scores, v)
    
    # Compute final output
    out_ptr = Out + bid_z * stride_oz + bid_h * stride_oh + bid_m * BLOCK_M * stride_om
    output = acc / sum_val[:, None]
    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_om +
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_on, output)
    
    # Store softmax statistics
    l_ptr = L + bid_z * H * N_CTX + bid_h * N_CTX + bid_m * BLOCK_M
    m_ptr = m + bid_z * H * N_CTX + bid_h * N_CTX + bid_m * BLOCK_M
    tl.store(l_ptr + tl.arange(0, BLOCK_M), sum_val)
    tl.store(m_ptr + tl.arange(0, BLOCK_M), max_val)

@triton.jit
def _bwd_preprocess(
    DO, Delta,
    stride_doz, stride_doh, stride_dom, stride_don,
    stride_dz, stride_dh, stride_dm,
    Z, H, N_CTX, BLOCK_M: tl.constexpr,
    L
):
    pid = tl.program_id(0)
    
    # Calculate indices
    bid_z = pid // (H * triton.cdiv(N_CTX, BLOCK_M))
    bid_h = (pid % (H * triton.cdiv(N_CTX, BLOCK_M))) // triton.cdiv(N_CTX, BLOCK_M)
    bid_m = (pid % (H * triton.cdiv(N_CTX, BLOCK_M))) % triton.cdiv(N_CTX, BLOCK_M)
    
    # Load L
    l_ptr = L + bid_z * H * N_CTX + bid_h * N_CTX + bid_m * BLOCK_M
    l = tl.load(l_ptr + tl.arange(0, BLOCK_M))
    
    # Load DO block
    do_ptr = DO + bid_z * stride_doz + bid_h * stride_doh + bid_m * BLOCK_M * stride_dom
    do = tl.load(do_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_dom +
                 tl.arange(0, BLOCK_M)[None, :] * stride_don)
    
    # Compute delta
    delta = tl.sum(do, 1) / l
    
    # Store delta
    delta_ptr = Delta + bid_z * stride_dz + bid_h * stride_dh + bid_m * BLOCK_M * stride_dm
    tl.store(delta_ptr + tl.arange(0, BLOCK_M), delta)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        # Save inputs for backward
        ctx.save_for_backward(q, k, v)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        
        # Extract dimensions
        Z, H, M, K = q.shape
        _, _, N, _ = k.shape
        
        # Allocate output
        out = torch.empty_like(q)
        L = torch.empty((Z, H, M), device=q.device, dtype=torch.float32)
        m = torch.empty_like(L)
        
        # Configure grid
        grid = (Z * H * triton.cdiv(M, 128),)
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, out, L, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            Z, H, M,
            BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=K,
            sm_scale=sm_scale,
            causal=causal,
            num_warps=4,
            num_stages=1
        )
        
        ctx.L = L
        ctx.m = m
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        L = ctx.L
        m = ctx.m
        
        Z, H, M, K = q.shape
        _, _, N, _ = k.shape
        
        # Allocate gradients
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty((Z, H, M), device=q.device, dtype=torch.float32)
        
        # Configure grid
        grid = (Z * H * triton.cdiv(M, 128),)
        
        # Launch preprocessing kernel
        _bwd_preprocess[grid](
            dout, delta,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            Z, H, M, BLOCK_M=128,
            L=L,
            num_warps=4
        )
        
        # Launch backward kernel (implementation not shown for brevity)
        # _bwd_kernel[grid](...)
        
        return dq, dk, dv, None, None

def attention(q, k, v, sm_scale, causal=False):
    """
    Compute scaled dot-product attention.
    q: (Z, H, M, K) - Query
    k: (Z, H, N, K) - Key
    v: (Z, H, N, K) - Value
    sm_scale: float - Scaling factor for dot product
    causal: bool - Apply causal mask
    """
    return _attention.apply(q, k, v, sm_scale, causal)
