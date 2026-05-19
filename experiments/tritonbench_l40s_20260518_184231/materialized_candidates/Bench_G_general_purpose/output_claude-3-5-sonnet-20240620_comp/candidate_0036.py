import triton
import triton.language as tl
import torch

@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qk, stride_kn, stride_km,
    stride_mn, stride_om, stride_on,
    N_CTX, sm_scale,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Load mask
    mask = tl.load(M + offs_m * stride_mn + offs_n * stride_mn)
    
    # Sliding window mask
    if WINDOW_SIZE > 0:
        m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        window_mask = tl.abs(m_idx[:, None] - n_idx[None, :]) <= WINDOW_SIZE
        mask = mask * window_mask

    # Iterate over k dimension
    for k in range(0, N_CTX, BLOCK_K):
        # Load Q and K blocks
        q = tl.load(Q + offs_m[:, None] * stride_qm + (k + offs_k)[None, :] * stride_qk)
        k = tl.load(K + (k + offs_k)[:, None] * stride_km + offs_n[None, :] * stride_kn)
        
        # Matrix multiplication
        acc += tl.dot(q, k)
    
    # Scale and apply mask
    acc = acc * sm_scale
    acc = acc + mask
    
    # Store result
    tl.store(Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc)

def get_score(q, k, mask=None, window_size=-1):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Reshape inputs
    q = q.reshape(-1, seq_len, d_head)
    k = k.reshape(-1, seq_len, d_head)
    
    # Create output tensor
    output = torch.empty((batch_size * n_heads, seq_len, seq_len), 
                        device=q.device, dtype=torch.float32)
    
    # Create default mask if none provided
    if mask is None:
        mask = torch.zeros((seq_len, seq_len), device=q.device, dtype=torch.float32)
    
    # Calculate scaling factor
    sm_scale = 1.0 / (d_head ** 0.5)
    
    # Grid configuration
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel with retry mechanism
    while True:
        try:
            _score_kernel[grid](
                q, k, mask, output,
                q.stride(0), q.stride(2),
                k.stride(2), k.stride(1),
                mask.stride(0), output.stride(0), output.stride(1),
                seq_len, sm_scale,
                window_size,
                BLOCK_M, BLOCK_N, BLOCK_K
            )
            break
        except triton.OutOfResources:
            # Reduce block sizes and retry
            BLOCK_M //= 2
            BLOCK_N //= 2
            if BLOCK_M < 32 or BLOCK_N < 32:
                raise RuntimeError("Block size too small, cannot proceed")
            
            grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
    
    # Reshape output back to original dimensions
    return output.reshape(batch_size, n_heads, seq_len, seq_len)
