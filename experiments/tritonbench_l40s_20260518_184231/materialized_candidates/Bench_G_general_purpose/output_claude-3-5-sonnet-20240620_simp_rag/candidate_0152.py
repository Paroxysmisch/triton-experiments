import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=4),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h,  # Pointers to tensors
    h0,  # Initial state (optional)
    T: tl.constexpr,  # Sequence length
    D: tl.constexpr,  # Hidden dimension
    BT: tl.constexpr,  # Block size for T
    BD: tl.constexpr,  # Block size for D
    USE_INITIAL_STATE: tl.constexpr,
):
    # Program ID
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    
    # Load block pointers
    k_ptr = k + pid_b * T * D + pid_t * BT * D + offs_d
    v_ptr = v + pid_b * T * D + pid_t * BT * D + offs_d
    h_ptr = h + pid_b * T * D + pid_t * BT * D + offs_d
    
    # Initialize accumulators
    h_acc = tl.zeros([BD], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE and pid_t == 0:
        h_acc += tl.load(h0 + pid_b * D + offs_d, mask=mask_d)
    
    # Main loop
    for i in range(BT):
        t = pid_t * BT + i
        mask = mask_d & (t < T)
        
        # Load inputs
        k_i = tl.load(k_ptr + i * D, mask=mask)
        v_i = tl.load(v_ptr + i * D, mask=mask)
        
        # Update accumulator
        h_acc = h_acc * tl.exp(k_i) + v_i
        
        # Store result
        tl.store(h_ptr + i * D, h_acc, mask=mask)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o,  # Pointers to tensors
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # Program ID
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Compute offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    
    # Process chunks
    for t in range(1, tl.cdiv(T, BT)):
        # Load pointers
        q_ptr = q + pid_b * T * D + t * BT * D + offs_d
        h_ptr = h + pid_b * T * D + (t-1) * BT * D + offs_d
        o_ptr = o + pid_b * T * D + t * BT * D + offs_d
        
        # Load previous hidden state
        h_prev = tl.load(h_ptr, mask=mask_d)
        
        # Load query and compute attention
        for i in range(BT):
            mask = mask_d & ((t * BT + i) < T)
            q_i = tl.load(q_ptr + i * D, mask=mask)
            o_i = q_i * h_prev
            tl.store(o_ptr + i * D, o_i, mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    dout, k, v, dh,  # Pointers to tensors
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # Program ID
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    
    # Initialize gradient accumulators
    dh_acc = tl.zeros([BD], dtype=tl.float32)
    
    # Process chunk in reverse
    for i in range(BT-1, -1, -1):
        t = pid_t * BT + i
        mask = mask_d & (t < T)
        
        # Load gradients and inputs
        dout_i = tl.load(dout + pid_b * T * D + t * D + offs_d, mask=mask)
        k_i = tl.load(k + pid_b * T * D + t * D + offs_d, mask=mask)
        
        # Compute gradients
        dh_acc = dh_acc * tl.exp(k_i) + dout_i
        tl.store(dh + pid_b * T * D + t * D + offs_d, dh_acc, mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    dout, q, k, v, h,
    dq, dk, dv,  # Output gradients
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # Program ID
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute offsets
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    
    # Process chunk
    for i in range(BT):
        t = pid_t * BT + i
        mask = mask_d & (t < T)
        
        # Load values
        dout_i = tl.load(dout + pid_b * T * D + t * D + offs_d, mask=mask)
        h_i = tl.load(h + pid_b * T * D + t * D + offs_d, mask=mask)
        
        # Compute gradients
        dq_i = dout_i * h_i
        dk_i = dout_i * h_i * tl.exp(k)
        dv_i = dout_i
        
        # Store gradients
        tl.store(dq + pid_b * T * D + t * D + offs_d, dq_i, mask=mask)
        tl.store(dk + pid_b * T * D + t * D + offs_d, dk_i, mask=mask)
        tl.store(dv + pid_b * T * D + t * D + offs_d, dv_i, mask=mask)
