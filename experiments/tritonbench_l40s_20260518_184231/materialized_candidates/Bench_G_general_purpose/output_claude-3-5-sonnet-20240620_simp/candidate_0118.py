import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_rwkv6_kernel(
    # Pointers to matrices
    r_ptr, k_ptr, v_ptr, w_ptr, u_ptr, o_ptr, state_ptr,
    # Matrix dimensions
    seq_len, hidden_dim,
    # Strides
    stride_seq_r, stride_h_r,
    stride_seq_k, stride_h_k,
    stride_seq_v, stride_h_v,
    stride_seq_w, stride_h_w,
    stride_seq_u, stride_h_u,
    stride_seq_o, stride_h_o,
    # Additional parameters
    scale: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute the sequence position
    seq_idx = pid
    
    # Handle bounds checking
    if seq_idx >= seq_len:
        return
        
    # Load the current state
    state = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if seq_idx > 0:
        state = tl.load(state_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Compute offsets for the current sequence position
    r_off = seq_idx * stride_seq_r + tl.arange(0, BLOCK_SIZE) * stride_h_r
    k_off = seq_idx * stride_seq_k + tl.arange(0, BLOCK_SIZE) * stride_h_k
    v_off = seq_idx * stride_seq_v + tl.arange(0, BLOCK_SIZE) * stride_h_v
    w_off = seq_idx * stride_seq_w + tl.arange(0, BLOCK_SIZE) * stride_h_w
    u_off = seq_idx * stride_seq_u + tl.arange(0, BLOCK_SIZE) * stride_h_u
    
    # Load inputs
    r = tl.load(r_ptr + r_off)
    k = tl.load(k_ptr + k_off)
    v = tl.load(v_ptr + v_off)
    w = tl.load(w_ptr + w_off)
    u = tl.load(u_ptr + u_off)
    
    # Compute attention
    state = state * tl.exp(w)  # Apply time-decay
    p = k + u  # Compute attention score
    exp_p = tl.exp(p)
    state = state + exp_p * v  # Update state
    
    # Compute output
    output = r * state * scale
    
    # Store output
    o_off = seq_idx * stride_seq_o + tl.arange(0, BLOCK_SIZE) * stride_h_o
    tl.store(o_ptr + o_off, output)
    
    # Store final state if this is the last sequence position
    if seq_idx == seq_len - 1:
        tl.store(state_ptr + tl.arange(0, BLOCK_SIZE), state)

def fused_recurrent_rwkv6(r, k, v, w, u, scale=1.0, initial_state=None, output_final_state=False):
    """
    Fused recurrent operation for RWKV-6 model.
    
    Args:
        r: Input tensor of shape [seq_len, hidden_dim]
        k: Input tensor of shape [seq_len, hidden_dim]
        v: Input tensor of shape [seq_len, hidden_dim]
        w: Input tensor of shape [seq_len, hidden_dim]
        u: Input tensor of shape [seq_len, hidden_dim]
        scale: Scaling factor for the output (default: 1.0)
        initial_state: Optional initial state tensor of shape [hidden_dim]
        output_final_state: Whether to return the final state
    
    Returns:
        o: Output tensor of shape [seq_len, hidden_dim]
        final_state: (Optional) Final state tensor of shape [hidden_dim]
    """
    seq_len, hidden_dim = r.shape
    
    # Allocate output tensor
    o = torch.empty_like(r)
    
    # Initialize or allocate state tensor
    if initial_state is None:
        state = torch.zeros(hidden_dim, device=r.device, dtype=r.dtype)
    else:
        state = initial_state.clone()
    
    # Define block size for the kernel
    BLOCK_SIZE = min(hidden_dim, 1024)
    
    # Launch kernel
    grid = (seq_len,)
    fused_recurrent_rwkv6_kernel[grid](
        r, k, v, w, u, o, state,
        seq_len, hidden_dim,
        r.stride(0), r.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        w.stride(0), w.stride(1),
        u.stride(0), u.stride(1),
        o.stride(0), o.stride(1),
        scale,
        BLOCK_SIZE,
    )
    
    if output_final_state:
        return o, state
    return o
