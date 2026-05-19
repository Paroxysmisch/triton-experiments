import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, output_ptr,
    # Matrix dimensions
    batch, heads, seq_len, dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(batch, BLOCK_SIZE)
    num_pid_h = heads
    num_pid_s = tl.cdiv(seq_len, BLOCK_SIZE)
    
    # Calculate batch/head/sequence indices
    pid_b = pid // (num_pid_h * num_pid_s)
    pid_h = (pid % (num_pid_h * num_pid_s)) // num_pid_s
    pid_s = (pid % (num_pid_h * num_pid_s)) % num_pid_s

    # Block pointers
    block_b = pid_b * BLOCK_SIZE
    block_s = pid_s * BLOCK_SIZE

    # Load query block
    q_block_ptr = q_ptr + block_b * stride_qb + pid_h * stride_qh + block_s * stride_qs
    q = tl.load(q_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qs + 
                tl.arange(0, dim)[None, :], mask=None)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE, dim], dtype=tl.float32)
    
    # Scale factor for attention scores
    scale = 1.0 / (dim ** 0.5)
    
    # Loop over key/value sequence length
    for k_idx in range(0, seq_len, BLOCK_SIZE):
        # Load key block
        k_block_ptr = k_ptr + block_b * stride_kb + pid_h * stride_kh + k_idx * stride_ks
        k = tl.load(k_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_ks + 
                    tl.arange(0, dim)[None, :], mask=None)
        
        # Load value block
        v_block_ptr = v_ptr + block_b * stride_vb + pid_h * stride_vh + k_idx * stride_vs
        v = tl.load(v_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_vs + 
                    tl.arange(0, dim)[None, :], mask=None)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * scale
        
        # Apply softmax
        scores = tl.softmax(scores)
        
        # Compute weighted sum
        acc += tl.dot(scores, v)
    
    # Write output
    output_block_ptr = (output_ptr + block_b * stride_ob + pid_h * stride_oh + 
                       block_s * stride_os)
    output_mask = (block_s + tl.arange(0, BLOCK_SIZE)) < seq_len
    tl.store(output_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_os + 
             tl.arange(0, dim)[None, :], acc, mask=output_mask[:, None])

def context_attention_fwd(q, k, v):
    """
    Forward pass for context attention.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, dim)
        k: Key tensor of shape (batch, heads, seq_len, dim)
        v: Value tensor of shape (batch, heads, seq_len, dim)
        
    Returns:
        Output tensor of shape (batch, heads, seq_len, dim)
    """
    batch, heads, seq_len, dim = q.shape
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Select block size based on GPU architecture
    BLOCK_SIZE = 32
    if torch.cuda.get_device_capability()[0] >= 7:
        BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (batch * heads * triton.cdiv(seq_len, BLOCK_SIZE),)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, output,
        batch, heads, seq_len, dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
