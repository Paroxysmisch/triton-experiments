import torch
import triton
import triton.language as tl

# Define the forward kernel for computing 'h'
@triton.jit
def chunk_retention_fwd_kernel_h(
    K, V, initial_state, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    b_h, final_state, NT, decay_fn, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize buffer
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load initial state if required
    if USE_INITIAL_STATE:
        h = tl.load(initial_state + offs)
    
    # Iterate over time dimension
    for t in range(NT):
        k_t = tl.load(K + t * BLOCK_SIZE + offs)
        v_t = tl.load(V + t * BLOCK_SIZE + offs)
        
        # Compute decay factors
        d_b, d_i = decay_fn(t)
        
        # Update buffer 'h'
        h = d_b * h + d_i * tl.dot(k_t, v_t)
        
        # Optionally store final state
        if STORE_FINAL_STATE and t == NT - 1:
            tl.store(final_state + offs, h)
    
    # Store result in 'b_h'
    tl.store(b_h + offs, h)

# Define the forward kernel for computing 'o'
@triton.jit
def chunk_retention_fwd_kernel_o(
    Q, K, V, H, b_o, b_s, decay_fn, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load blocks
    q = tl.load(Q + offs)
    k = tl.load(K + offs)
    v = tl.load(V + offs)
    h = tl.load(H + offs)
    
    # Compute decay factor
    d_i = decay_fn(0)  # Example: decay function usage
    
    # Compute contributions
    o = tl.dot(q, k) * v * d_i
    s = tl.dot(q, h) * d_i
    
    # Store results
    tl.store(b_o + offs, o)
    tl.store(b_s + offs, s)

# Define backward kernel for computing 'dh'
@triton.jit
def chunk_retention_bwd_kernel_dh(
    dO, dH, NT, decay_fn, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize buffer for dh
    dh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate backwards over time steps
    for t in range(NT - 1, -1, -1):
        d_o_t = tl.load(dO + t * BLOCK_SIZE + offs)
        
        # Accumulate gradient contributions
        dh += d_o_t * decay_fn(t)
    
    # Store result in 'dH'
    tl.store(dH + offs, dh)

# Define backward kernel for computing 'dq', 'dk', 'dv'
@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    dO, Q, K, V, dQ, dK, dV, decay_fn, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load blocks
    q = tl.load(Q + offs)
    k = tl.load(K + offs)
    v = tl.load(V + offs)
    d_o = tl.load(dO + offs)
    
    # Compute decay factor
    d_i = decay_fn(0)  # Example: decay function usage
    
    # Compute gradients
    dq = d_o * k * d_i
    dk = d_o * q * d_i
    dv = d_o * v * d_i
    
    # Store gradients
    tl.store(dQ + offs, dq)
    tl.store(dK + offs, dk)
    tl.store(dV + offs, dv)

# Define the custom PyTorch autograd function
class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, use_initial_state=False, store_final_state=False):
        # Prepare inputs and outputs
        NT, BLOCK_SIZE = q.shape[0], q.shape[1]
        h = torch.zeros_like(q)
        o = torch.zeros_like(q)
        s = torch.zeros_like(q)
        final_state = torch.zeros_like(q) if store_final_state else None
        
        # Launch Triton kernels
        chunk_retention_fwd_kernel_h[(NT,)](k, v, initial_state, use_initial_state, store_final_state, h, final_state, NT, lambda t: (0.9, 0.1), BLOCK_SIZE=BLOCK_SIZE)
        chunk_retention_fwd_kernel_o[(NT,)](q, k, v, h, o, s, lambda t: 0.1, BLOCK_SIZE=BLOCK_SIZE)
        
        # Save context for backward
        ctx.save_for_backward(q, k, v, h)
        
        return o, final_state

    @staticmethod
    def backward(ctx, do, dfinal_state=None):
        q, k, v, h = ctx.saved_tensors
        NT, BLOCK_SIZE = q.shape[0], q.shape[1]
        
        # Prepare gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dh = torch.zeros_like(h)
        
        # Launch Triton backward kernels
        chunk_retention_bwd_kernel_dh[(NT,)](do, dh, NT, lambda t: 0.1, BLOCK_SIZE=BLOCK_SIZE)
        chunk_retention_bwd_kernel_dqkv[(NT,)](do, q, k, v, dq, dk, dv, lambda t: 0.1, BLOCK_SIZE=BLOCK_SIZE)
        
        return dq, dk, dv, None, None, None

# Define the chunk retention function as a user interface
def chunk_retention(q, k, v, initial_state=None, use_initial_state=False, store_final_state=False):
    return ChunkRetentionFunction.apply(q, k, v, initial_state, use_initial_state, store_final_state)
