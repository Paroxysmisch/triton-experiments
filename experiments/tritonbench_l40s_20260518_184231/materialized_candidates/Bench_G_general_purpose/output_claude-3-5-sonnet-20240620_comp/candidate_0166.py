import torch
import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr, h_ptr,
    # Matrix dimensions
    batch, heads, T, d,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_hb, stride_hh, stride_ht,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE: tl.constexpr, IFCOND: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(T, BLOCK_M)
    num_pid_n = tl.cdiv(T, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_in_group
    
    # Initialize pointers to Q, K, V, O
    offs_m = (pid % num_pid_m) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = (pid % num_pid_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    q_ptrs = q_ptr + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    
    # Load Q, K, V
    q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[:, None] < T, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0)
    
    # Compute attention scores
    scale = 1.0 / tl.sqrt(float(d))
    scores = tl.dot(q, k.transpose()) * scale
    
    # Apply softmax
    scores = scores - tl.max(scores, 1)[:, None]
    scores = tl.exp(scores)
    normalizer = tl.sum(scores, 1)[:, None]
    scores = scores / normalizer
    
    # Compute attention output
    o = tl.dot(scores, v)
    
    # Store output
    offs_o = offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptr + offs_o, o, mask=offs_m[:, None] < T)
    
    # Store attention scores if required
    if STORE:
        h_ptrs = h_ptr + offs_m[:, None] * stride_ht + offs_n[None, :] * 1
        tl.store(h_ptrs, scores, mask=(offs_m[:, None] < T) & (offs_n[None, :] < T))

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # Dimensions
        batch, heads, T, d = q.shape
        
        # Allocate output
        o = torch.empty_like(q)
        h = torch.empty((batch, heads, T, T), device=q.device, dtype=q.dtype)
        
        # Configure kernel parameters
        grid = (batch * heads,)
        num_warps = 4
        num_stages = 2
        
        # Launch kernel
        attention_fwd_kernel[grid](
            q, k, v, o, h,
            batch, heads, T, d,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=32,
            STORE=True, IFCOND=False,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        return o

# Example usage
def attention_forward(q, k, v):
    return AttentionFunction.apply(q, k, v)
