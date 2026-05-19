import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def fused_recurrent_fwd_kernel(
    # Input tensors
    q_ptr, k_ptr, v_ptr,
    beta_ptr,
    # Output tensors
    o_ptr, h_ptr,
    # Stride information
    stride_q_b, stride_q_h, stride_q_t, stride_q_k,
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    stride_v_b, stride_v_h, stride_v_t, stride_v_v,
    # Parameters
    B, H, T, K, V,
    BK: tl.constexpr, BV: tl.constexpr,
    scale: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_BETA: tl.constexpr,
    REVERSE: tl.constexpr,
):
    # Parallelize over batch-heads and features
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    
    # Offsets for feature blocks
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    
    # Initialize pointers
    q_offset = i_bh * stride_q_b + (T-1)*stride_q_t * REVERSE
    k_offset = i_bh * stride_k_b + (T-1)*stride_k_t * REVERSE
    v_offset = i_bh * stride_v_b + (T-1)*stride_v_t * REVERSE
    o_offset = i_bh * stride_q_b + (T-1)*stride_q_t * REVERSE
    
    # Initialize hidden state
    h = tl.zeros((BV, BK), dtype=tl.float32)
    if USE_INITIAL_STATE:
        h0_ptr = h_ptr + i_bh * K * V
        h0_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
        h = tl.load(h0_ptr + offs_v[:, None]*K + offs_k[None, :], mask=h0_mask, other=0.0)
    
    for t in range(T):
        # Load current timestep data
        q = tl.load(q_ptr + q_offset + offs_k, mask=offs_k < K, other=0.0)
        k = tl.load(k_ptr + k_offset + offs_k, mask=offs_k < K, other=0.0)
        v = tl.load(v_ptr + v_offset + offs_v, mask=offs_v < V, other=0.0)
        
        # Apply scaling and beta
        q_scaled = q * scale
        if HAS_BETA:
            beta = tl.load(beta_ptr + i_bh * stride_q_b + offs_k, mask=offs_k < K)
            q_scaled *= beta
        
        # Update hidden state
        h += tl.dot(v[:, None], k[None, :])
        o = tl.dot(h, q_scaled[:, None])
        
        # Store output
        tl.store(o_ptr + o_offset + offs_v, o.to(o_ptr.dtype.element_ty), mask=offs_v < V)
        
        # Update offsets
        step = -1 if REVERSE else 1
        q_offset += step * stride_q_t
        k_offset += step * stride_k_t
        v_offset += step * stride_v_t
        o_offset += step * stride_q_t
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        final_h_ptr = h_ptr + (B*H + i_bh) * K * V
        h_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
        tl.store(final_h_ptr + offs_v[:, None]*K + offs_k[None, :], h, mask=h_mask)

@triton.jit
def fused_recurrent_bwd_kernel(
    # Input tensors
    q_ptr, k_ptr, v_ptr, beta_ptr,
    do_ptr,
    # Output gradients
    dq_ptr, dk_ptr, dv_ptr, dbeta_ptr,
    # Stride information
    stride_q_b, stride_q_h, stride_q_t, stride_q_k,
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    stride_v_b, stride_v_h, stride_v_t, stride_v_v,
    # Parameters
    B, H, T, K, V,
    BK: tl.constexpr, BV: tl.constexpr,
    scale: tl.constexpr,
    HAS_BETA: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    
    # Initialize pointers for reverse pass
    step = 1 if REVERSE else -1
    offset = 0 if REVERSE else (T-1)*stride_q_t
    
    # Initialize gradient accumulators
    dh = tl.zeros((BV, BK), dtype=tl.float32)
    
    for t in range(T):
        # Load current timestep data
        q = tl.load(q_ptr + offset + offs_k, mask=offs_k < K)
        k = tl.load(k_ptr + offset + offs_k, mask=offs_k < K)
        v = tl.load(v_ptr + offset + offs_v, mask=offs_v < V)
        do = tl.load(do_ptr + offset + offs_v, mask=offs_v < V)
        
        # Compute gradients
        dq = tl.dot(dh, k[:, None]).to(tl.float32) * scale
        if HAS_BETA:
            beta = tl.load(beta_ptr + i_bh * stride_q_b + offs_k)
            dq *= beta
            dbeta = tl.sum(dq * q, axis=1)
            tl.store(dbeta_ptr + offset + offs_k, dbeta, mask=offs_k < K)
        
        dk = tl.dot(dh.T, v[:, None])
        dv = tl.dot(dh, q[:, None])
        
        # Update hidden state gradient
        dh = dh * (1.0 - REVERSE*2)  # Adjust direction
        
        # Store gradients
        tl.store(dq_ptr + offset + offs_k, dq, mask=offs_k < K)
        tl.store(dk_ptr + offset + offs_k, dk, mask=offs_k < K)
        tl.store(dv_ptr + offset + offs_v, dv, mask=offs_v < V)
        
        offset += step * stride_q_t

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, beta, scale, initial_state, reverse):
        # Save dimensions and parameters
        B, H, T, K = q.shape
        V = v.shape[-1]
        ctx.scale = scale or (K ** -0.5)
        ctx.reverse = reverse
        
        # Allocate output tensor
        o = torch.empty_like(v)
        final_state = torch.empty(B, H, K, V, device=q.device) if initial_state is None else None
        
        # Configure kernel launch parameters
        BK, BV = 32, 32  # Tune these based on hardware
        grid = (triton.cdiv(V, BV), B * H)
        
        fused_recurrent_fwd_kernel[grid](
            q, k, v, beta, o, final_state,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            B, H, T, K, V, BK, BV,
            scale=ctx.scale,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            HAS_BETA=beta is not None,
            REVERSE=reverse,
        )
        
        ctx.save_for_backward(q, k, v, beta, final_state)
        return o, final_state

    @staticmethod
    @custom_bwd
    def backward(ctx, do, d_final_state):
        q, k, v, beta, final_state = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        
        # Initialize gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dbeta = torch.zeros_like(beta) if beta is not None else None
        
        BK, BV = 32, 32
        grid = (triton.cdiv(V, BV), B * H)
        
        fused_recurrent_bwd_kernel[grid](
            q, k, v, beta, do, dq, dk, dv, dbeta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            B, H, T, K, V, BK, BV,
            scale=ctx.scale,
            HAS_BETA=beta is not None,
            REVERSE=ctx.reverse,
        )
        
        return dq, dk, dv, dbeta, None, None, None

def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    reverse: bool = False
) -> torch.Tensor:
    """
    Fused recurrent computation using delta rule.
    
    Args:
        q: Query tensor [B, H, T, K]
        k: Key tensor [B, H, T, K]
        v: Value tensor [B, H, T, V]
        beta: Optional scaling tensor [B, H, K] or scalar
        scale: Overall scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        reverse: Process sequence in reverse
        
    Returns:
        output: [B, H, T, V]
        final_state: [B, H, K, V]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    
    # Convert scalar beta to tensor if needed
    if beta is not None and not isinstance(beta, torch.Tensor):
        beta = torch.full((q.shape[0], q.shape[1], q.shape[-1]), beta, device=q.device)
    
    o, final_state = FusedRecurrentFunction.apply(
        q, k, v, beta, scale, initial_state, reverse
    )
    
    return o, final_state
