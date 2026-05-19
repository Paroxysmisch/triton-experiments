import torch
import triton
import triton.language as tl

@triton.jit
def fused_pairwise_distance_normalize_kernel(
    # Pointers to matrices
    x1_ptr, x2_ptr, output_ptr,
    # Matrix dimensions
    batch_size, feat_size, seq_len_1, seq_len_2,
    # Parameters
    p_norm, eps_norm, eps_distance,
    # Strides for the tensors
    stride_x1_batch, stride_x1_seq, stride_x1_feat,
    stride_x2_batch, stride_x2_seq, stride_x2_feat,
    stride_out_batch, stride_out_seq1, stride_out_seq2,
    BLOCK_SIZE: tl.constexpr
):
    # Compute pid and index
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len_1 * seq_len_2)
    rem = pid % (seq_len_1 * seq_len_2)
    seq1_idx = rem // seq_len_2
    seq2_idx = rem % seq_len_2

    # Compute memory offsets
    x1_off = batch_idx * stride_x1_batch + seq1_idx * stride_x1_seq
    x2_off = batch_idx * stride_x2_batch + seq2_idx * stride_x2_seq
    
    # Load feature vectors
    offs_feat = tl.arange(0, BLOCK_SIZE)
    mask_feat = offs_feat < feat_size
    
    x1_feats = tl.load(x1_ptr + x1_off + offs_feat * stride_x1_feat, mask=mask_feat, other=0.0)
    x2_feats = tl.load(x2_ptr + x2_off + offs_feat * stride_x2_feat, mask=mask_feat, other=0.0)

    # Normalize x1
    x1_norm = tl.sum(tl.abs(x1_feats) ** p_norm, axis=0)
    x1_norm = tl.maximum(x1_norm ** (1.0 / p_norm), eps_norm)
    x1_feats = x1_feats / x1_norm

    # Normalize x2
    x2_norm = tl.sum(tl.abs(x2_feats) ** p_norm, axis=0)
    x2_norm = tl.maximum(x2_norm ** (1.0 / p_norm), eps_norm)
    x2_feats = x2_feats / x2_norm

    # Compute distance
    diff = x1_feats - x2_feats
    dist = tl.sum(tl.abs(diff) ** p_norm, axis=0)
    dist = tl.maximum(dist ** (1.0 / p_norm), eps_distance)

    # Store result
    output_off = (batch_idx * stride_out_batch + 
                 seq1_idx * stride_out_seq1 + 
                 seq2_idx * stride_out_seq2)
    tl.store(output_ptr + output_off, dist)

def fused_pairwise_distance_normalize(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12,
    eps_distance: float = 1e-6,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Computes pairwise distance between normalized input tensors.
    
    Args:
        x1 (Tensor): First input tensor of shape (B, N, D)
        x2 (Tensor): Second input tensor of shape (B, M, D)
        p_norm (float): The exponent value in the norm for normalization
        eps_norm (float): Small value to avoid division by zero during normalization
        eps_distance (float): Small value to avoid division by zero in distance calculation
        keepdim (bool): Whether to keep the last dimension in output
    
    Returns:
        Tensor: Pairwise distances of shape (B, N, M) or (B, N, M, 1) if keepdim=True
    """
    assert x1.dim() == 3 and x2.dim() == 3, "Input tensors must be 3-dimensional"
    assert x1.size(0) == x2.size(0), "Batch sizes must match"
    assert x1.size(2) == x2.size(2), "Feature dimensions must match"
    
    batch_size = x1.size(0)
    seq_len_1 = x1.size(1)
    seq_len_2 = x2.size(1)
    feat_size = x1.size(2)
    
    # Ensure contiguous tensors
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Prepare output tensor
    output = torch.empty((batch_size, seq_len_1, seq_len_2), 
                        device=x1.device, dtype=x1.dtype)
    
    # Configure grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(feat_size)
    grid = (batch_size * seq_len_1 * seq_len_2,)
    
    # Launch kernel
    fused_pairwise_distance_normalize_kernel[grid](
        x1_ptr=x1, x2_ptr=x2, output_ptr=output,
        batch_size=batch_size, feat_size=feat_size,
        seq_len_1=seq_len_1, seq_len_2=seq_len_2,
        p_norm=p_norm, eps_norm=eps_norm, eps_distance=eps_distance,
        stride_x1_batch=x1.stride(0), stride_x1_seq=x1.stride(1), stride_x1_feat=x1.stride(2),
        stride_x2_batch=x2.stride(0), stride_x2_seq=x2.stride(1), stride_x2_feat=x2.stride(2),
        stride_out_batch=output.stride(0), stride_out_seq1=output.stride(1), 
        stride_out_seq2=output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if keepdim:
        output = output.unsqueeze(-1)
    
    return output
