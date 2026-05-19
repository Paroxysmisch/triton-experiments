import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr,
                stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
                stride_vb, stride_vh, stride_vn, stride_om, stride_oh, stride_ob,
                B, H, M, N, D, scale, BLOCK_SIZE: tl.constexpr):
    # Get the program index for this kernel
    pid = tl.program_id(axis=0)
    
    # Compute batch and head indices
    batch_id = pid // H
    head_id = pid % H
    
    # Compute the offset for this batch and head
    Q_offset = batch_id * stride_qb + head_id * stride_qh
    K_offset = batch_id * stride_kb + head_id * stride_kh
    V_offset = batch_id * stride_vb + head_id * stride_vh
    Out_offset = batch_id * stride_ob + head_id * stride_oh
    
    # Create pointers for this block
    Q = tl.load(Q_ptr + Q_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qm)
    K = tl.load(K_ptr + K_offset + tl.arange(0, BLOCK_SIZE)[None, :] * stride_kn)
    V = tl.load(V_ptr + V_offset + tl.arange(0, BLOCK_SIZE)[None, :] * stride_vn)
    
    # Compute QK^T
    QK = tl.dot(Q, K, trans_b=True) * scale
    
    # Apply softmax to get attention scores
    attn_scores = tl.softmax(QK, axis=1)
    
    # Compute output by weighting values with attention scores
    Out = tl.dot(attn_scores, V)
    
    # Store the result
    tl.store(Out_ptr + Out_offset + tl.arange(0, BLOCK_SIZE)[:, None] * stride_om, Out)

def context_attention_fwd(Q, K, V, scale):
    # Extract dimensions
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    
    # Allocate output tensor
    Out = torch.empty((B, H, M, D), device=Q.device, dtype=Q.dtype)
    
    # Define block size
    BLOCK_SIZE = 128  # Adjust based on your hardware
    
    # Define grid size
    grid = (B * H,)
    
    # Launch the Triton kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(2), Out.stride(1), Out.stride(0),
        B, H, M, N, D, scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Out
