import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr, z_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Scale for attention
    scale,
    # Normalization options
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = group_id
    pid_n = pid % num_pid_in_group

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers to Q, K, V
    q_block_ptr = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_block_ptr = k_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kh)
    v_block_ptr = v_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vh)

    # Load Q, K, V
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # Compute attention scores
    scores = tl.dot(q, k.transpose(1, 0)) * scale
    
    # Apply softmax
    scores = scores - tl.max(scores, 1)[:, None]
    scores = tl.exp(scores)
    z = tl.sum(scores, 1)[:, None]
    scores = scores / z

    # Compute output
    o = tl.dot(scores, v)
    
    # Write back output and normalizer
    o_block_ptr = o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_oh)
    z_block_ptr = z_ptr + offs_m
    tl.store(o_block_ptr, o)
    tl.store(z_block_ptr, z.squeeze(1))

@triton.jit
def parallel_rebased_bwd_kernel(
    # Pointers to matrices
    dout_ptr, q_ptr, k_ptr, v_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    o_ptr, z_ptr,
    # Matrix dimensions and strides (same as forward)
    batch_size, seq_len, num_heads, head_dim,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Scale for attention
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = group_id
    pid_n = pid % num_pid_in_group

    # Block pointers (similar to forward pass)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load necessary matrices
    dout = tl.load(dout_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_oh))
    q = tl.load(q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh))
    k = tl.load(k_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kh))
    v = tl.load(v_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vh))
    z = tl.load(z_ptr + offs_m)

    # Compute gradients
    # dV = attention_probs^T * dout
    dv = tl.dot(scores.transpose(1, 0), dout)
    
    # dK = dout * V^T * attention_probs
    dk = tl.dot(dout, v.transpose(1, 0)) * scores
    
    # dQ = dout * K * attention_probs
    dq = tl.dot(dout, k) * scores

    # Scale gradients
    dq = dq * scale
    dk = dk * scale
    
    # Store gradients
    tl.store(dq_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh), dq)
    tl.store(dk_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kh), dk)
    tl.store(dv_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vh), dv)

class ParallelRebasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=1.0, normalize=True):
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.normalize = normalize
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        # Output tensors
        o = torch.empty_like(q)
        z = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)
        
        # Grid for parallel execution
        grid = (batch_size * num_heads * triton.cdiv(seq_len, 32),)
        
        # Launch kernel
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z,
            batch_size, seq_len, num_heads, head_dim,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            scale,
            BLOCK_M=32, BLOCK_N=32, BLOCK_DMODEL=head_dim,
        )
        
        return o, z

    @staticmethod
    def backward(ctx, grad_output, grad_z):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Grid for parallel execution
        grid = (batch_size * num_heads * triton.cdiv(seq_len, 32),)
        
        # Launch backward kernel
        parallel_rebased_bwd_kernel[grid](
            grad_output, q, k, v, dq, dk, dv,
            batch_size, seq_len, num_heads, head_dim,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            scale,
            BLOCK_M=32, BLOCK_N=32, BLOCK_DMODEL=head_dim,
        )
        
        return dq, dk, dv, None, None

def parallel_rebased(q, k, v, scale=1.0, normalize=True):
    """
    Wrapper function for parallel rebased attention.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        scale: Scaling factor for attention scores
        normalize: Whether to apply normalization
    
    Returns:
        Tuple of (output, normalizer)
    """
    return ParallelRebasedFunction.apply(q, k, v, scale, normalize)
