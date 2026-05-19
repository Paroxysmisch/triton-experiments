import triton
import triton.language as tl
import torch

@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    BT, BK, BV,
    q_stride_b, q_stride_t, q_stride_k,
    k_stride_b, k_stride_t, k_stride_k,
    v_stride_b, v_stride_t, v_stride_v,
    h_stride_b, h_stride_t, h_stride_v,
    o_stride_b, o_stride_t, o_stride_v,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr
):
    # Block index
    block_idx = tl.program_id(0)
    
    # Load q, k, v, h blocks
    q = tl.load(q_ptr + block_idx * q_stride_t, mask=True)
    k = tl.load(k_ptr + block_idx * k_stride_t, mask=True)
    v = tl.load(v_ptr + block_idx * v_stride_t, mask=True)
    h = tl.load(h_ptr + block_idx * h_stride_t, mask=True)
    
    # Compute dot product q * k^T
    qk = tl.dot(q, k, trans_b=True)
    
    # Apply scaling and exponential
    qk_scaled = qk * SCALE
    qk_exp = tl.exp(qk_scaled)
    
    # Compute attention scores and apply to v
    attn_scores = tl.dot(qk_exp, v)
    
    # Combine with auxiliary tensor h
    o = attn_scores + h
    
    # Store result in output
    tl.store(o_ptr + block_idx * o_stride_t, o, mask=True)

def chunk_fwd_o_fn(q, k, v, h, BT, BK, BV, scale):
    # Ensure inputs are on GPU
    assert q.is_cuda and k.is_cuda and v.is_cuda and h.is_cuda
    
    # Get strides and pointers
    q_ptr, k_ptr, v_ptr, h_ptr = q.data_ptr(), k.data_ptr(), v.data_ptr(), h.data_ptr()
    o = torch.empty_like(h)  # Output tensor
    o_ptr = o.data_ptr()
    
    q_stride_b, q_stride_t, q_stride_k = q.stride()
    k_stride_b, k_stride_t, k_stride_k = k.stride()
    v_stride_b, v_stride_t, v_stride_v = v.stride()
    h_stride_b, h_stride_t, h_stride_v = h.stride()
    o_stride_b, o_stride_t, o_stride_v = o.stride()
    
    # Define grid
    grid = (BT, )
    
    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
        BT, BK, BV,
        q_stride_b, q_stride_t, q_stride_k,
        k_stride_b, k_stride_t, k_stride_k,
        v_stride_b, v_stride_t, v_stride_v,
        h_stride_b, h_stride_t, h_stride_v,
        o_stride_b, o_stride_t, o_stride_v,
        BLOCK_SIZE=128,  # Example block size
        SCALE=scale
    )
    
    return o
