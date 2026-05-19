import triton
import triton.language as tl
import torch

@triton.jit
def chunk_retention_fwd_kernel_h(
    # Pointers to tensors
    k_ptr, v_ptr, h_ptr,
    # Dimensions
    B, T, H, D,
    BT,  # Chunk size
    # Strides
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_h_b, stride_h_t, stride_h_h, stride_h_d,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = pid // H
    h = pid % H
    
    # Block start index
    b = bid // ((T + BT - 1) // BT)
    t_start = (bid % ((T + BT - 1) // BT)) * BT
    
    # Initialize hidden state
    h_prev = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Offsets for loading k and v
    offs_d = tl.arange(0, BLOCK_SIZE)
    
    # Process chunk
    for t in range(t_start, min(t_start + BT, T)):
        # Load k and v
        k_offs = b * stride_k_b + t * stride_k_t + h * stride_k_h + offs_d * stride_k_d
        v_offs = b * stride_v_b + t * stride_v_t + h * stride_v_h + offs_d * stride_v_d
        
        k = tl.load(k_ptr + k_offs, mask=offs_d < D)
        v = tl.load(v_ptr + v_offs, mask=offs_d < D)
        
        # Update hidden state
        h_curr = k * v + h_prev
        h_prev = h_curr
        
        # Store hidden state
        h_offs = b * stride_h_b + t * stride_h_t + h * stride_h_h + offs_d * stride_h_d
        tl.store(h_ptr + h_offs, h_curr, mask=offs_d < D)

# Python wrapper for the hidden state kernel
def chunk_fwd_h_fn(k, v, BT):
    B, T, H, D = k.shape
    
    # Allocate output
    h = torch.empty_like(k)
    
    # Launch kernel
    grid = (B * ((T + BT - 1) // BT) * H,)
    chunk_retention_fwd_kernel_h[grid](
        k, v, h,
        B, T, H, D, BT,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        BLOCK_SIZE=min(D, 1024),
    )
    return h

@triton.jit
def chunk_retention_fwd_kernel_o(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Dimensions
    B, T, H, D,
    BT,  # Chunk size
    scale,  # Scaling factor
    # Strides
    stride_q_b, stride_q_t, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_o_b, stride_o_t, stride_o_h, stride_o_d,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = pid // H
    h = pid % H
    
    # Block start index
    b = bid // ((T + BT - 1) // BT)
    t_start = (bid % ((T + BT - 1) // BT)) * BT
    
    # Offsets for loading/storing
    offs_d = tl.arange(0, BLOCK_SIZE)
    
    # Process chunk
    for t in range(t_start, min(t_start + BT, T)):
        # Load q
        q_offs = b * stride_q_b + t * stride_q_t + h * stride_q_h + offs_d * stride_q_d
        q = tl.load(q_ptr + q_offs, mask=offs_d < D)
        
        # Initialize output
        o = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute attention for current position
        for j in range(t_start, t + 1):
            k_offs = b * stride_k_b + j * stride_k_t + h * stride_k_h + offs_d * stride_k_d
            v_offs = b * stride_v_b + j * stride_v_t + h * stride_v_h + offs_d * stride_v_d
            
            k = tl.load(k_ptr + k_offs, mask=offs_d < D)
            v = tl.load(v_ptr + v_offs, mask=offs_d < D)
            
            # Compute attention score
            score = tl.sum(q * k) * scale
            
            # Apply decay based on distance
            decay = tl.exp(-(t - j))
            
            # Update output
            o += v * (score * decay)
        
        # Store output
        o_offs = b * stride_o_b + t * stride_o_t + h * stride_o_h + offs_d * stride_o_d
        tl.store(o_ptr + o_offs, o, mask=offs_d < D)

# Python wrapper for the output kernel
def chunk_fwd_o_fn(q, k, v, scale, BT):
    B, T, H, D = q.shape
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Launch kernel
    grid = (B * ((T + BT - 1) // BT) * H,)
    chunk_retention_fwd_kernel_o[grid](
        q, k, v, o,
        B, T, H, D, BT, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BLOCK_SIZE=min(D, 1024),
    )
    return o
