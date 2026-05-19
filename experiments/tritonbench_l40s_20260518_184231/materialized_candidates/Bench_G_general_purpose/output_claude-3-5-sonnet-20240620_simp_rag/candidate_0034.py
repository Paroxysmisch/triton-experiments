import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2(
    # Pointers to matrices
    prob_ptr, value_ptr, out_ptr,
    # Matrix dimensions
    batch, heads, seqlen, dim,
    # Strides for the different matrices
    stride_prob_b, stride_prob_h, stride_prob_s,
    stride_value_b, stride_value_h, stride_value_s, stride_value_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    # Compute offsets
    offs_seq = pid_seq * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_dim = tl.arange(0, BLOCK_SIZE)
    
    # Initialize pointers
    prob_offset = (
        pid_batch * stride_prob_b +
        pid_head * stride_prob_h +
        pid_seq * BLOCK_SIZE * stride_prob_s
    )
    value_offset = (
        pid_batch * stride_value_b +
        pid_head * stride_value_h
    )
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate through sequence length
    for start_s in range(0, seqlen, BLOCK_SIZE):
        # Load probabilities
        p = tl.load(
            prob_ptr + prob_offset + start_s * stride_prob_s + tl.arange(0, BLOCK_SIZE),
            mask=offs_seq[:, None] < seqlen,
            other=0.0
        )
        
        # Load values
        v = tl.load(
            value_ptr + value_offset + start_s * stride_value_s + offs_dim[None, :],
            mask=start_s + offs_dim[None, :] < dim,
            other=0.0
        )
        
        # Compute attention
        acc += tl.dot(p, v)
    
    # Write output
    out_offset = (
        pid_batch * stride_out_b +
        pid_head * stride_out_h +
        pid_seq * BLOCK_SIZE * stride_out_s
    )
    
    tl.store(
        out_ptr + out_offset + offs_seq[:, None] * stride_out_s + offs_dim[None, :],
        acc,
        mask=(offs_seq[:, None] < seqlen) & (offs_dim[None, :] < dim)
    )

def token_att_fwd2(prob, value):
    """
    Wrapper function for token attention forward pass
    
    Args:
        prob: attention probability tensor [batch, heads, seqlen, seqlen]
        value: value tensor [batch, heads, seqlen, dim]
    
    Returns:
        output tensor [batch, heads, seqlen, dim]
    """
    batch, heads, seqlen, _ = prob.shape
    dim = value.shape[-1]
    
    # Allocate output
    output = torch.empty((batch, heads, seqlen, dim), 
                        device=prob.device, dtype=value.dtype)
    
    # Configure block size based on GPU capability
    BLOCK_SIZE = 32
    
    # Configure grid
    grid = (batch, heads, triton.cdiv(seqlen, BLOCK_SIZE))
    
    # Launch kernel
    _fwd_kernel_token_att2[grid](
        prob, value, output,
        batch, heads, seqlen, dim,
        prob.stride(0), prob.stride(1), prob.stride(2),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return output
