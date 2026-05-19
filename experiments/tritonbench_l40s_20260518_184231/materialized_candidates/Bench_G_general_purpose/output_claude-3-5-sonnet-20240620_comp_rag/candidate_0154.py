import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h, initial_state,
    final_state,
    T: tl.constexpr, D: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    # Get program ID
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask = offs_d < D
    
    # Initialize buffer
    b_h = tl.zeros([BD], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE and pid_t == 0:
        b_h += tl.load(initial_state + pid_b * D + offs_d, mask=mask)
    
    # Main computation loop
    for t in range(BT):
        # Current time step
        curr_t = pid_t * BT + t
        if curr_t >= T:
            break
            
        # Load k and v blocks
        k_ptr = k + (pid_b * T + curr_t) * D + offs_d
        v_ptr = v + (pid_b * T + curr_t) * D + offs_d
        
        b_k = tl.load(k_ptr, mask=mask)
        b_v = tl.load(v_ptr, mask=mask)
        
        # Compute decay factors
        d_b = tl.exp(-1.0 / D)
        d_i = 1.0 - d_b
        
        # Update hidden state
        b_h = d_b * b_h + d_i * (b_k * b_v)
        
        # Store result
        h_ptr = h + (pid_b * T + curr_t) * D + offs_d
        tl.store(h_ptr, b_h, mask=mask)
    
    # Store final state if needed
    if STORE_FINAL_STATE and pid_t == (T // BT - 1):
        fs_ptr = final_state + pid_b * D + offs_d
        tl.store(fs_ptr, b_h, mask=mask)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o,
    T: tl.constexpr, D: tl.constexpr,
    BT: tl.constexpr, BD: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask = offs_d < D
    
    # Initialize output buffers
    b_o = tl.zeros([BD], dtype=tl.float32)
    b_s = tl.zeros([1], dtype=tl.float32)
    
    for t in range(BT):
        curr_t = pid_t * BT + t
        if curr_t >= T:
            break
            
        # Load blocks
        q_ptr = q + (pid_b * T + curr_t) * D + offs_d
        k_ptr = k + (pid_b * T + curr_t) * D + offs_d
        v_ptr = v + (pid_b * T + curr_t) * D + offs_d
        h_ptr = h + (pid_b * T + curr_t) * D + offs_d
        
        b_q = tl.load(q_ptr, mask=mask)
        b_k = tl.load(k_ptr, mask=mask)
        b_v = tl.load(v_ptr, mask=mask)
        b_h = tl.load(h_ptr, mask=mask)
        
        # Compute attention scores
        d_i = 1.0 - tl.exp(-1.0 / D)
        score = tl.sum(b_q * b_k * d_i)
        b_s += score
        
        # Update output
        b_o += score * b_v + b_h
        
    # Store final output
    o_ptr = o + (pid_b * T + pid_t * BT) * D + offs_d
    tl.store(o_ptr, b_o / (b_s + 1e-6), mask=mask)

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, store_final_state=False):
        B, H, T, D = q.shape
        BT = min(128, T)
        BD = min(128, triton.next_power_of_2(D))
        
        # Allocate output tensors
        h = torch.empty_like(q)
        o = torch.empty_like(q)
        final_state = torch.empty((B, H, D), device=q.device) if store_final_state else None
        
        # Launch kernels
        grid = (triton.cdiv(D, BD), triton.cdiv(T, BT), B * H)
        chunk_retention_fwd_kernel_h[grid](
            k, v, h, initial_state, final_state,
            T, D, BT, BD,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=store_final_state
        )
        
        chunk_retention_fwd_kernel_o[grid](
            q, k, v, h, o,
            T, D, BT, BD
        )
        
        ctx.save_for_backward(q, k, v, h, o)
        return o, final_state if store_final_state else None

    @staticmethod
    def backward(ctx, grad_o, grad_final_state=None):
        q, k, v, h, o = ctx.saved_tensors
        # Backward implementation would go here
        # For brevity, returning None for all gradients
        return None, None, None, None, None

def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    store_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies chunk-wise retention mechanism to input tensors.
    
    Args:
        q: Query tensor of shape (B, H, T, D)
        k: Key tensor of shape (B, H, T, D)
        v: Value tensor of shape (B, H, T, D)
        initial_state: Optional initial hidden state of shape (B, H, D)
        store_final_state: Whether to return the final hidden state
        
    Returns:
        Tuple of (output tensor, final state if requested else None)
    """
    return ChunkRetentionFunction.apply(q, k, v, initial_state, store_final_state)
