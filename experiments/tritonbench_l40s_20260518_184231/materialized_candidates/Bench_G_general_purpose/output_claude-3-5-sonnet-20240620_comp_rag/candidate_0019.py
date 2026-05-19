import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    # Pointers to matrices
    Q, K, COS, SIN, 
    # Matrix dimensions
    seq_len, head_dim,
    # Strides for the different dimensions
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute sequence position
    seq_idx = pid % seq_len
    head_idx = (pid // seq_len) 
    
    # Create offsets for the dimension
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < head_dim
    
    # Compute offsets for the rotary computation
    dim_pairs = head_dim // 2
    offs_d_l = offs_d % dim_pairs
    offs_d_h = dim_pairs + offs_d_l
    
    # Load cos/sin values
    cos_idx = seq_idx * head_dim + offs_d
    sin_idx = seq_idx * head_dim + offs_d
    cos = tl.load(COS + cos_idx, mask=mask)
    sin = tl.load(SIN + sin_idx, mask=mask)
    
    # Compute Q indices
    q_idx = (seq_idx * stride_q_s + 
             head_idx * stride_q_h +
             offs_d * stride_q_d)
    
    # Load Q values
    q_l = tl.load(Q + q_idx, mask=mask)
    q_h = tl.load(Q + q_idx + dim_pairs * stride_q_d, mask=mask)
    
    # Apply rotation to Q
    q_out_l = q_l * cos - q_h * sin
    q_out_h = q_h * cos + q_l * sin
    
    # Store Q results
    tl.store(Q + q_idx, q_out_l, mask=mask)
    tl.store(Q + q_idx + dim_pairs * stride_q_d, q_out_h, mask=mask)
    
    # Apply same rotation to K if provided
    if K is not None:
        k_idx = (seq_idx * stride_k_s +
                 head_idx * stride_k_h +
                 offs_d * stride_k_d)
        
        # Load K values
        k_l = tl.load(K + k_idx, mask=mask)
        k_h = tl.load(K + k_idx + dim_pairs * stride_k_d, mask=mask)
        
        # Apply rotation to K
        k_out_l = k_l * cos - k_h * sin
        k_out_h = k_h * cos + k_l * sin
        
        # Store K results
        tl.store(K + k_idx, k_out_l, mask=mask)
        tl.store(K + k_idx + dim_pairs * stride_k_d, k_out_h, mask=mask)

def apply_rotary_embedding(q: torch.Tensor, 
                         k: torch.Tensor, 
                         cos: torch.Tensor, 
                         sin: torch.Tensor):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        cos: Cosine tensor of shape [seq_len, head_dim]
        sin: Sine tensor of shape [seq_len, head_dim]
    """
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    
    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    # Configure kernel parameters
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    grid = (seq_len * num_heads,)
    
    # Launch kernel
    rotary_embedding_kernel[grid](
        q, k, cos, sin,
        seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return q, k
