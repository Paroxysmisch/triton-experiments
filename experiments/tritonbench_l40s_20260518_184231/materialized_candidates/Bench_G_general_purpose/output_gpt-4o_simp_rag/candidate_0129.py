import triton
import triton.language as tl
import torch

@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, h_ptr,
    initial_state_ptr, final_state_ptr,
    B, H, N, D,  # Dimensions: Batch, Head, Time, Feature
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_dh, stride_dn, stride_dd,
    stride_vnewh, stride_vnewn, stride_vnewd,
    stride_hh, stride_hn, stride_hd,
    has_initial_state: tl.constexpr,
    has_final_state: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Pointers
    k_ptrs = k_ptr + offs_h[:, None] * stride_kh + offs_n[None, :] * stride_kn
    v_ptrs = v_ptr + offs_h[:, None] * stride_vh + offs_n[None, :] * stride_vn
    d_ptrs = d_ptr + offs_h[:, None] * stride_dh + offs_n[None, :] * stride_dn
    
    # Load data
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    d = tl.load(d_ptrs)
    
    # Compute block-wise dot product
    v_new = tl.dot(k, v)
    
    # Update v_new with d
    v_new += d
    
    # Store the result
    v_new_ptrs = v_new_ptr + offs_h[:, None] * stride_vnewh + offs_n[None, :] * stride_vnewn
    tl.store(v_new_ptrs, v_new)
    
    # Handle initial and final states
    if has_initial_state:
        initial_state_ptrs = initial_state_ptr + offs_h * stride_hh
        tl.store(initial_state_ptrs, v_new)
    
    if has_final_state:
        final_state_ptrs = final_state_ptr + offs_h * stride_hh
        tl.store(final_state_ptrs, v_new)

def chunk_fwd_h_fn(k, v, d, B, H, N, D, initial_state=None, final_state=None):
    # Initialize output tensor
    v_new = torch.empty((B, H, N, D), dtype=k.dtype, device=k.device)
    
    # Determine grid and block sizes
    BLOCK_SIZE = 128
    grid = (H, N // BLOCK_SIZE)
    
    # Launch kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, initial_state, final_state,
        B, H, N, D,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d.stride(0), d.stride(1), d.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        initial_state.stride(0) if initial_state is not None else 0,
        final_state.stride(0) if final_state is not None else 0,
        initial_state is not None,
        final_state is not None,
        BLOCK_SIZE
    )
    
    return v_new
