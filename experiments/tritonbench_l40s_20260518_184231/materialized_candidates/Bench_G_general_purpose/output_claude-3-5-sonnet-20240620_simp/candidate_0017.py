import torch
import triton
import triton.language as tl

# Constants for block sizes and stages
BLOCK_M = 128
BLOCK_N = 128
HEAD_DIM = 64
STAGE = 3

@triton.jit
def _attn_fwd_inner(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    
    # Initialize offsets
    offs_m = (pid % num_pid_m) * BLOCK_M
    offs_n = (pid // num_pid_m) * BLOCK_N
    
    # Load Q block
    q_ptrs = Q + offs_m * stride_qm
    k_ptrs = K + offs_n * stride_kn
    v_ptrs = V + offs_n * stride_vn
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, HEAD_DIM, BLOCK_N):
        # Load Q and K blocks
        q = tl.load(q_ptrs + k * stride_qk)
        k = tl.load(k_ptrs + k * stride_kk)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * sm_scale
        
        # Apply softmax
        scores = tl.softmax(scores)
        
        # Load V block and compute output
        v = tl.load(v_ptrs + k * stride_vk)
        acc += tl.dot(scores, v)
    
    # Store results
    out_ptrs = Out + offs_m * stride_om
    tl.store(out_ptrs, acc)

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Calculate number of blocks and total programs
    num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    num_program = num_block_m * num_block_n
    
    # Launch inner kernel for each block
    for pid in range(num_program):
        _attn_fwd_inner(
            Q, K, V, sm_scale, Out,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vn, stride_vk,
            stride_oz, stride_oh, stride_om, stride_on,
            Z, H, N_CTX,
            BLOCK_M, BLOCK_N, STAGE,
        )

def forward(q, k, v, sm_scale):
    """
    Forward pass of multi-head attention.
    
    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        sm_scale: Softmax scaling factor
        
    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Initialize output tensor
    output = torch.empty_like(q)
    
    # Calculate grid size
    grid = (
        triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),
    )
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, sm_scale, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, num_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, STAGE=STAGE,
    )
    
    return output
