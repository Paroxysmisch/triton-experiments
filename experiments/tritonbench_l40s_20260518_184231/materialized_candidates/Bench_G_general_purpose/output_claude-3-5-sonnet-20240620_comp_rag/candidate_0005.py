import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V,           # Query, Key, Value tensors
    sm_scale,          # Softmax scaling factor
    Out,              # Output tensor
    Tmp,              # Temporary storage
    stride_qz, stride_qh, stride_qm, stride_qk,  # Strides for Q
    stride_kz, stride_kh, stride_kn, stride_kk,  # Strides for K
    stride_vz, stride_vh, stride_vn, stride_vk,  # Strides for V
    Z, H,             # Batch size and number of heads
    N_CTX,            # Sequence length
    BLOCK: tl.constexpr,  # Block size (e.g., 64)
):
    # Program ID gives us the block we're computing
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK)
    num_pid_n = tl.cdiv(N_CTX, BLOCK)
    
    # Initialize pointers to Q, K, V
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    
    # Load Q block
    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    max_val = tl.zeros([BLOCK], dtype=tl.float32) - float("inf")
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Compute attention scores
    for block_n in range(0, num_pid_n):
        k_idx = block_n * BLOCK + offs_n
        k = tl.load(K + k_idx[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v = tl.load(V + k_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        
        # Compute Q @ K.T
        scores = tl.dot(q, tl.trans(k))
        scores = scores * sm_scale
        
        # Apply causal mask
        scores = tl.where(offs_m[:, None] >= k_idx[None, :], scores, float("-inf"))
        
        # Compute softmax
        scores_max = tl.max(scores, 1)
        scores = scores - scores_max[:, None]
        scores = tl.exp(scores)
        scores_sum = tl.sum(scores, 1)
        scores = scores / scores_sum[:, None]
        
        # Update output
        acc += tl.dot(scores, v)
    
    # Store output
    tl.store(Out + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, acc)

@triton.jit
def _bwd_kernel(
    Q, K, V, DOut,    # Forward inputs and output gradient
    DQ, DK, DV,       # Output gradients for Q, K, V
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Similar structure to forward pass, but computing gradients
    offs_m = pid * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    
    # Load Q block
    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    dout = tl.load(DOut + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    
    # Initialize gradient accumulators
    dq_acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    dk_acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    
    # Compute gradients
    for block_n in range(0, tl.cdiv(N_CTX, BLOCK)):
        k_idx = block_n * BLOCK + offs_n
        k = tl.load(K + k_idx[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v = tl.load(V + k_idx[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        
        # Forward pass computations (needed for backward)
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        scores = tl.where(offs_m[:, None] >= k_idx[None, :], scores, float("-inf"))
        scores = tl.exp(scores - tl.max(scores, 1)[:, None])
        scores = scores / tl.sum(scores, 1)[:, None]
        
        # Gradient computations
        dv_acc += tl.dot(tl.trans(scores), dout)
        ds = tl.dot(dout, tl.trans(v))
        ds = ds * scores
        
        dq_acc += tl.dot(ds, k) * sm_scale
        dk_acc += tl.dot(tl.trans(ds), q) * sm_scale
    
    # Store gradients
    tl.store(DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, dq_acc)
    tl.store(DK + offs_m[:, None] * stride_kn + offs_k[None, :] * stride_kk, dk_acc)
    tl.store(DV + offs_m[:, None] * stride_vn + offs_k[None, :] * stride_vk, dv_acc)
