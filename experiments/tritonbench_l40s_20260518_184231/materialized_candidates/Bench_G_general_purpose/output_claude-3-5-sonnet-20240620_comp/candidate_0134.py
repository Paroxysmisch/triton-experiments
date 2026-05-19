import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Matrix dimensions
    B_Start_Loc, B_Seqlen,  # Batch information
    H, # Number of heads
    Lq, Lk, # Query length, Key length
    D, # Head dimension
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm, stride_qd,  # Q strides
    stride_kb, stride_kh, stride_kn, stride_kd,  # K strides
    stride_vb, stride_vh, stride_vn, stride_vd,  # V strides
    stride_ob, stride_oh, stride_om, stride_od,  # Output strides
    # Scale for attention scores
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr  # Blocking parameters
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Lq, BLOCK_M)
    num_pid_n = tl.cdiv(Lk, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    num_groups = B_Seqlen.shape[0] * H
    if group_id >= num_groups:
        return
    
    # Batch and head index
    batch_id = group_id // H
    head_id = group_id % H
    
    # Initialize offsets
    offs_m = (pid % num_pid_m) * BLOCK_M
    offs_n = ((pid % num_pid_in_group) // num_pid_m) * BLOCK_N
    
    # Load query block
    q_ptrs = Q + (batch_id * stride_qb + head_id * stride_qh + offs_m * stride_qm)
    k_ptrs = K + (batch_id * stride_kb + head_id * stride_kh + offs_n * stride_kn)
    
    # Initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q = tl.load(q_ptrs + offs_d[None, :] * stride_qd, mask=offs_d[None, :] < D, other=0.0)
    k = tl.load(k_ptrs + offs_d[None, :] * stride_kd, mask=offs_d[None, :] < D, other=0.0)
    
    # Compute attention scores
    scores = tl.dot(q, k.transpose())
    scores = scores * sm_scale
    
    # Compute softmax
    scores = scores - tl.max(scores, 1)[:, None]
    scores = tl.exp(scores)
    normalizer = tl.sum(scores, 1)[:, None]
    scores = scores / normalizer
    
    # Load values and compute output
    v_ptrs = V + (batch_id * stride_vb + head_id * stride_vh + offs_n * stride_vn)
    v = tl.load(v_ptrs + offs_d[None, :] * stride_vd, mask=offs_d[None, :] < D, other=0.0)
    
    output = tl.dot(scores, v)
    
    # Write output
    out_ptrs = Out + (batch_id * stride_ob + head_id * stride_oh + offs_m * stride_om)
    tl.store(out_ptrs + offs_d[None, :] * stride_od, output, mask=offs_d[None, :] < D)

def context_attention_fwd(q, k, v, B_Start_Loc, B_Seqlen):
    batch_size, num_heads, max_seqlen, head_dim = q.shape
    
    # Compute scale for attention scores
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Calculate strides
    stride_qb, stride_qh, stride_qm, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = v.stride()
    stride_ob, stride_oh, stride_om, stride_od = output.stride()
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = head_dim
    
    # Compute grid size
    grid = (batch_size * num_heads * triton.cdiv(max_seqlen, BLOCK_M),)
    
    # Number of warps depends on the key length
    num_warps = 4 if max_seqlen <= 128 else 8
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, output,
        B_Start_Loc, B_Seqlen,
        num_heads, max_seqlen, max_seqlen, head_dim,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
    
    return output
