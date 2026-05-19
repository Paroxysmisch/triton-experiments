import torch
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    # Tensor pointers
    Q, K, COS, SIN, Q_EMB, K_EMB,
    # Dimensions and strides
    seq_len, stride_qs, stride_qh, stride_qd,
    stride_ks, stride_kh, stride_kd,
    stride_cos, stride_sin,
    # Kernel parameters
    HEAD_DIM: tl.constexpr, Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Position offsets
    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offsets < seq_len
    
    # Head offsets
    head_offsets = pid_head * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    is_q = head_offsets < Q_HEAD_NUM
    head_mask = head_offsets < (Q_HEAD_NUM + K_HEAD_NUM)
    
    # Feature dimension offsets
    dim_offsets = tl.arange(0, BLOCK_DMODEL)
    half_dim = BLOCK_DMODEL // 2
    dim_mask = dim_offsets < half_dim
    
    # Combined masks
    mask = tl.logical_and(seq_mask[:, None, None], 
                         tl.logical_and(head_mask[None, :, None], dim_mask[None, None, :]))
    
    # Load cosine/sine values
    cos_offsets = seq_offsets[:, None] * stride_cos + dim_offsets[None, :]
    sin_offsets = seq_offsets[:, None] * stride_sin + dim_offsets[None, :]
    cos = tl.load(COS + cos_offsets, mask=seq_mask[:, None] & dim_mask[None, :])
    sin = tl.load(SIN + sin_offsets, mask=seq_mask[:, None] & dim_mask[None, :])

    # Process queries
    q_offsets = (seq_offsets[:, None, None] * stride_qs + 
                head_offsets[None, :, None] * stride_qh +
                dim_offsets[None, None, :] * stride_qd)
    q = tl.load(Q + q_offsets, mask=mask & is_q[None, :, None], other=0)
    
    # Process keys
    k_offsets = (seq_offsets[:, None, None] * stride_ks + 
                (head_offsets - Q_HEAD_NUM)[None, :, None] * stride_kh +
                dim_offsets[None, None, :] * stride_kd)
    k = tl.load(K + k_offsets, mask=mask & (~is_q)[None, :, None], other=0)

    # Split dimensions for rotary transformation
    q0 = q[..., :half_dim]
    q1 = q[..., half_dim:]
    k0 = k[..., :half_dim]
    k1 = k[..., half_dim:]

    # Apply rotary embeddings
    q_emb0 = q0 * cos - q1 * sin
    q_emb1 = q1 * cos + q0 * sin
    k_emb0 = k0 * cos - k1 * sin
    k_emb1 = k1 * cos + k0 * sin

    # Store results
    tl.store(Q_EMB + q_offsets[..., :half_dim], q_emb0, mask=mask & is_q[None, :, None])
    tl.store(Q_EMB + q_offsets[..., half_dim:], q_emb1, mask=mask & is_q[None, :, None])
    tl.store(K_EMB + k_offsets[..., :half_dim], k_emb0, mask=mask & (~is_q)[None, :, None])
    tl.store(K_EMB + k_offsets[..., half_dim:], k_emb1, mask=mask & (~is_q)[None, :, None])

@triton.jit
def fused_rotary_embedding_kernel_v2(
    Q, K_CACHE, BLOCK_TABLES, KV_LENGTHS,
    COS, SIN, Q_EMB,
    stride_qs, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_bt, stride_bl,
    seq_len, HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_TABLE_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    # Load block table for current batch
    block_table = BLOCK_TABLES + pid_batch * stride_bt
    blocks = tl.load(block_table + tl.arange(0, BLOCK_TABLE_SIZE) * stride_bl)
    
    # Get current sequence position
    seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    valid_seq = seq_offsets < seq_len
    
    # Get current length and block info
    curr_len = tl.load(KV_LENGTHS + pid_batch)
    block_idx = (curr_len + seq_offsets) // BLOCK_SEQ
    block_offset = (curr_len + seq_offsets) % BLOCK_SEQ
    
    # Load relevant blocks from block table
    physical_blocks = tl.load(block_table + block_idx * stride_bl)
    
    # Calculate cache offsets
    cache_offsets = (physical_blocks * BLOCK_SEQ + block_offset) * stride_kt + \
                    pid_head * stride_kh + \
                    tl.arange(0, BLOCK_DMODEL) * stride_kd
    
    # Load cos/sine values
    cos_offsets = seq_offsets * stride_cos + tl.arange(0, BLOCK_DMODEL//2)
    sin_offsets = seq_offsets * stride_sin + tl.arange(0, BLOCK_DMODEL//2)
    cos = tl.load(COS + cos_offsets, mask=valid_seq[:, None])
    sin = tl.load(SIN + sin_offsets, mask=valid_seq[:, None])

    # Process queries
    q_offsets = seq_offsets * stride_qs + pid_head * stride_qh + tl.arange(0, BLOCK_DMODEL) * stride_qd
    q = tl.load(Q + q_offsets, mask=valid_seq[:, None])
    
    # Split and rotate query
    q0, q1 = tl.split(q, BLOCK_DMODEL//2, 1)
    q_emb0 = q0 * cos - q1 * sin
    q_emb1 = q1 * cos + q0 * sin
    tl.store(Q_EMB + q_offsets, tl.cat(q_emb0, q_emb1, 1), mask=valid_seq[:, None])

    # Process and store keys in cache
    k_offsets = cache_offsets + (pid_head % K_HEAD_NUM) * stride_kh
    k = tl.load(K_CACHE + k_offsets, mask=valid_seq[:, None])
    k0, k1 = tl.split(k, BLOCK_DMODEL//2, 1)
    k_emb0 = k0 * cos - k1 * sin
    k_emb1 = k1 * cos + k0 * sin
    tl.store(K_CACHE + k_offsets, tl.cat(k_emb0, k_emb1, 1), mask=valid_seq[:, None])

def rotary_embedding(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    q_emb: torch.Tensor = None,
    k_emb: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    block_tables: torch.Tensor = None,
    kv_lengths: torch.Tensor = None
):
    # Initial validation and setup
    assert q.dim() == 3 and k.dim() == 3, "Inputs must be 3D tensors"
    assert cos.shape == sin.shape, "Cos and Sin must have same shape"
    
    # Setup output tensors
    q_emb = q if q_emb is None else q_emb
    k_emb = k if k_emb is None else k_emb
    
    if k_cache is None:
        # Standard rotary embedding without cache
        BLOCK_SEQ = 16
        BLOCK_HEAD = 4
        BLOCK_DMODEL = 64
        
        grid = (
            triton.cdiv(q.size(0), BLOCK_SEQ),
            triton.cdiv(q.size(1) + k.size(1), BLOCK_HEAD),
        )
        
        rotary_embedding_kernel[grid](
            q, k, cos, sin, q_emb, k_emb,
            seq_len=q.size(0),
            stride_qs=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
            stride_ks=k.stride(0), stride_kh=k.stride(1), stride_kd=k.stride(2),
            stride_cos=cos.stride(0), stride_sin=sin.stride(0),
            HEAD_DIM=q.size(2), Q_HEAD_NUM=q.size(1),
            BLOCK_SEQ=BLOCK_SEQ, BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=4, num_stages=2
        )
    else:
        # Fused implementation with KV cache
        BLOCK_SEQ = 32
        BLOCK_HEAD = 2
        BLOCK_DMODEL = 128
        
        grid = (
            block_tables.size(0),  # batch size
            triton.cdiv(q.size(0), BLOCK_SEQ),
            q.size(1)  # num heads
        )
        
        fused_rotary_embedding_kernel_v2[grid](
            q, k_cache, block_tables, kv_lengths,
            cos, sin, q_emb,
            stride_qs=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
            stride_kt=k_cache.stride(0), stride_kh=k_cache.stride(1), stride_kd=k_cache.stride(2),
            stride_bt=block_tables.stride(0), stride_bl=block_tables.stride(1),
            seq_len=q.size(0), HEAD_DIM=q.size(2),
            BLOCK_SEQ=BLOCK_SEQ, BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_TABLE_SIZE=block_tables.size(1),
            num_warps=8, num_stages=4
        )
    
    return q_emb, k_emb if k_cache is None else k_cache

   q_emb, k_emb = rotary_embedding(q, k, cos, sin)
   
   q_emb, k_cache = rotary_embedding(
       q, k, cos, sin,
       k_cache=cache,
       block_tables=block_tables,
       kv_lengths=seq_lengths
   )
