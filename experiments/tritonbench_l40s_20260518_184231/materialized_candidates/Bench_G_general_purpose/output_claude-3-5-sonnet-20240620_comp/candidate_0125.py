import torch
import triton
import triton.language as tl

# Constants for block sizes
BTL = 128  # Block size for sequence length dimension
BTS = 32   # Block size for sequence dimension
BK = 32    # Block size for key dimension
BV = 32    # Block size for value dimension

@triton.jit
def parallel_rebased_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr, z_ptr,
    # Matrix dimensions
    batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Options
    use_scale: tl.constexpr,
    use_normalize: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len_q, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len_kv, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, num_pid_m - first_pid_m)
    pid_m = (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load query block
    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qh)
    
    # Load key block
    k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kh)
    
    # Matrix multiply
    acc += tl.dot(q, k)
    
    # Scale if requested
    if use_scale:
        acc = acc * (1.0 / tl.sqrt(float(head_dim)))
    
    # Normalize if requested
    if use_normalize:
        # Compute softmax
        max_val = tl.max(acc, 1)
        exp_val = tl.exp(acc - max_val[:, None])
        sum_val = tl.sum(exp_val, 1)
        acc = exp_val / sum_val[:, None]
        
        # Store normalization factors
        tl.store(z_ptr + offs_m, sum_val)
    
    # Load value block and compute output
    v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vh)
    o = tl.dot(acc, v)
    
    # Store output
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_oh, o)

@triton.jit
def _parallel_rebased_bwd_dq(
    dq_ptr, do_ptr, k_ptr, acc_ptr,
    batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
    stride_dqb, stride_dqh, stride_dqm,
    stride_dob, stride_doh, stride_dom,
    stride_kb, stride_kh, stride_kn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Similar block pointer setup as forward pass
    offs_m = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) % seq_len_q
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load blocks
    do = tl.load(do_ptr + offs_m[:, None] * stride_dom)
    k = tl.load(k_ptr + offs_n[None, :] * stride_kn)
    acc = tl.load(acc_ptr + offs_m[:, None] * stride_dqm + offs_n[None, :])
    
    # Compute gradients
    dq = tl.dot(acc, k)
    
    # Store gradients
    tl.store(dq_ptr + offs_m[:, None] * stride_dqm, dq)

@triton.jit
def _parallel_rebased_bwd_dkv(
    dk_ptr, dv_ptr, q_ptr, acc_ptr,
    batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
    stride_dkb, stride_dkh, stride_dkn,
    stride_dvb, stride_dvh, stride_dvn,
    stride_qb, stride_qh, stride_qm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Block pointer setup
    offs_m = (pid * BLOCK_M + tl.arange(0, BLOCK_M)) % seq_len_kv
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load blocks
    q = tl.load(q_ptr + offs_n[None, :] * stride_qm)
    acc = tl.load(acc_ptr + offs_n[None, :] * stride_qm + offs_m[:, None])
    
    # Compute gradients
    dk = tl.dot(acc.transpose(), q)
    dv = tl.dot(acc, q)
    
    # Store gradients
    tl.store(dk_ptr + offs_m[:, None] * stride_dkn, dk)
    tl.store(dv_ptr + offs_m[:, None] * stride_dvn, dv)

class ParallelRebasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, use_scale=True, use_normalize=True):
        # Save inputs for backward pass
        ctx.save_for_backward(q, k, v)
        ctx.use_scale = use_scale
        ctx.use_normalize = use_normalize
        
        # Get dimensions
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_kv, _ = k.shape
        
        # Allocate output
        o = torch.empty_like(q)
        z = torch.empty((batch_size, num_heads, seq_len_q), device=q.device)
        
        # Launch kernel
        grid = (batch_size * num_heads * triton.cdiv(seq_len_q, BTL),)
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z,
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            use_scale, use_normalize,
            BLOCK_M=BTL, BLOCK_N=BTS, BLOCK_K=BK
        )
        
        return o, z if use_normalize else o

    @staticmethod
    def backward(ctx, do, dz=None):
        q, k, v = ctx.saved_tensors
        
        # Get dimensions
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_kv, _ = k.shape
        
        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch backward kernels
        grid_q = (batch_size * num_heads * triton.cdiv(seq_len_q, BTL),)
        _parallel_rebased_bwd_dq[grid_q](
            dq, do, k, None,  # None for acc_ptr as we don't need it for this example
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            dq.stride(0), dq.stride(1), dq.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            BLOCK_M=BTL, BLOCK_N=BK
        )
        
        grid_kv = (batch_size * num_heads * triton.cdiv(seq_len_kv, BTL),)
        _parallel_rebased_bwd_dkv[grid_kv](
            dk, dv, q, None,  # None for acc_ptr as we don't need it for this example
            batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            q.stride(0), q.stride(1), q.stride(2),
            BLOCK_M=BTL, BLOCK_N=BK
        )
        
        return dq, dk, dv, None, None

def parallel_rebased(q, k, v, use_scale=True, use_normalize=True, return_both=False):
    """
    User-facing API for the parallel rebased attention mechanism.
    
    Args:
        q, k, v: Query, key and value tensors
        use_scale: Whether to apply scaling to attention scores
        use_normalize: Whether to apply normalization
        return_both: Whether to return both output and normalization factor
    
    Returns:
        Attention output tensor(s)
    """
    assert q.shape[-1] <= 128, "Feature dimension must not exceed 128"
    out = ParallelRebasedFunction.apply(q, k, v, use_scale, use_normalize)
    return out if return_both else out[0]
