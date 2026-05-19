import triton
import triton.language as tl
import torch

# Define the Triton kernel for the attention forward pass
@triton.jit
def attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_hb, stride_hh, stride_ht, stride_hd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T, D, SCALE, IFCOND: tl.constexpr, STORE: tl.constexpr
):
    # Define block sizes
    start_q = tl.program_id(0)
    start_h = tl.program_id(1)
    
    # Initialize block tensors
    b_h = tl.zeros((D, D), dtype=tl.float32)
    
    # Loop over sequence length
    for start_t in range(0, T, tl.cdiv(T, D)):
        # Set memory pointers
        p_q = q_ptr + start_q * stride_qb + start_h * stride_qh + start_t * stride_qt
        p_k = k_ptr + start_q * stride_kb + start_h * stride_kh + start_t * stride_kt
        p_v = v_ptr + start_q * stride_vb + start_h * stride_vh + start_t * stride_vt
        
        # Load blocks of query, key, value
        b_q = tl.load(p_q, mask=start_t < T)
        b_k = tl.load(p_k, mask=start_t < T)
        b_v = tl.load(p_v, mask=start_t < T)
        
        # Compute scaled dot-product attention scores
        b_s = tl.dot(b_q, b_k.T) * SCALE
        
        # Apply attention scores to value
        b_o = tl.dot(b_s, b_v)
        
        # Update intermediate tensor b_h based on IFCOND
        if IFCOND:
            b_h = tl.where(b_s > 0, b_h + b_o, b_h)
        else:
            b_h += b_o
        
        # Optionally store intermediate results
        if STORE:
            p_h = h_ptr + start_q * stride_hb + start_h * stride_hh + start_t * stride_ht
            tl.store(p_h, b_h, mask=start_t < T)
    
    # Store final output
    p_o = o_ptr + start_q * stride_ob + start_h * stride_oh + start_t * stride_ot
    tl.store(p_o, b_h, mask=start_t < T)

# Define the wrapper class
class AttentionFunction:
    def __init__(self):
        pass

    def forward(self, q, k, v, IFCOND=False, STORE=False):
        # Extract dimensions
        B, H, T, D = q.shape
        
        # Initialize output and intermediate storage
        o = torch.empty((B, H, T, D), device=q.device, dtype=q.dtype)
        h = torch.empty((B, H, T, D), device=q.device, dtype=q.dtype)
        
        # Set scaling factor
        SCALE = 1.0 / (D ** 0.5)
        
        # Launch kernel
        grid = (B, H)
        num_warps = 4
        num_stages = 2
        
        attention_fwd_kernel[grid](
            q, k, v, h, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            h.stride(0), h.stride(1), h.stride(2), h.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, T, D, SCALE, IFCOND, STORE,
            num_warps=num_warps, num_stages=num_stages
        )
        
        return o

# Example usage
# q, k, v are input tensors with shape [B, H, T, D]
attention_fn = AttentionFunction()
output = attention_fn.forward(q, k, v)
