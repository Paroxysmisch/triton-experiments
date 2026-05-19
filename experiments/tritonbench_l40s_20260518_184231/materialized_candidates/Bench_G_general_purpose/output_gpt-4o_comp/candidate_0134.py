import triton
import triton.language as tl
import torch

# Constants for block sizes
BLOCK_M = 128  # Number of queries processed in parallel
BLOCK_DMODEL = 64  # Dimensionality of each head
BLOCK_N = 128  # Number of keys processed in parallel

@triton.jit
def _fwd_kernel(Q, K, V, Out, sm_scale, B_Start_Loc, B_Seqlen, 
                stride_qz, stride_qh, stride_qm, stride_qd, 
                stride_kz, stride_kh, stride_kn, stride_kd, 
                stride_vz, stride_vh, stride_vn, stride_vd, 
                stride_oz, stride_oh, stride_om, stride_od, 
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    
    # Program IDs
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Offsets for Q, K, V
    offset_qm = pid_m * BLOCK_M
    offset_kn = pid_m * BLOCK_N

    # Load Q, K, V blocks
    Q_block = tl.load(Q + pid_z * stride_qz + pid_h * stride_qh + offset_qm * stride_qm, 
                      mask=offset_qm < B_Seqlen[pid_z])
    K_block = tl.load(K + pid_z * stride_kz + pid_h * stride_kh + offset_kn * stride_kn)
    V_block = tl.load(V + pid_z * stride_vz + pid_h * stride_vh + offset_kn * stride_vn)
    
    # Compute attention scores
    scores = tl.dot(Q_block, K_block, trans_b=True)
    scores = scores * sm_scale
    
    # Apply softmax
    max_scores = tl.max(scores, axis=1)
    scores = scores - max_scores[:, None]
    exp_scores = tl.exp(scores)
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    softmax_scores = exp_scores / sum_exp_scores[:, None]
    
    # Compute output
    Out_block = tl.dot(softmax_scores, V_block)
    
    # Store the result
    tl.store(Out + pid_z * stride_oz + pid_h * stride_oh + offset_qm * stride_om, Out_block)

# Wrapper function
def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen):
    # Compute sm_scale
    sm_scale = 1.0 / (Q.shape[-1] ** 0.5)
    
    # Determine grid size
    batch_size, num_heads, _, d_model = Q.shape
    grid = (batch_size, num_heads, (B_Seqlen.max() + BLOCK_M - 1) // BLOCK_M)
    
    # Strides
    stride_qz, stride_qh, stride_qm, stride_qd = Q.stride()
    stride_kz, stride_kh, stride_kn, stride_kd = K.stride()
    stride_vz, stride_vh, stride_vn, stride_vd = V.stride()
    stride_oz, stride_oh, stride_om, stride_od = Q.stride()  # Assuming Out has the same shape as Q
    
    # Allocate output tensor
    Out = torch.empty_like(Q)
    
    # Launch kernel
    _fwd_kernel[grid](
        Q, K, V, Out, sm_scale, B_Start_Loc, B_Seqlen,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N
    )
    
    return Out
