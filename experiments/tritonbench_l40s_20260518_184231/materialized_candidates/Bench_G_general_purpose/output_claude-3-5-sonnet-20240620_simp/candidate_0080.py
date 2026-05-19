import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    # Pointers to matrices
    Prob_ptr, V_ptr, Out_ptr,
    # Matrix dimensions
    batch_size, head_count, seq_len, head_dim,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Computes attention-like weighted sum: Out = Prob @ V
    Prob: (batch_size, head_count, seq_len, seq_len)
    V: (batch_size, head_count, seq_len, head_dim)
    Out: (batch_size, head_count, seq_len, head_dim)
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and head indices
    batch_id = pid // (head_count * triton.cdiv(seq_len, BLOCK_M))
    head_id = (pid % (head_count * triton.cdiv(seq_len, BLOCK_M))) // triton.cdiv(seq_len, BLOCK_M)
    seq_id = (pid % triton.cdiv(seq_len, BLOCK_M)) * BLOCK_M

    # Initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Iterate over blocks
    for start_n in range(0, seq_len, BLOCK_N):
        # Compute probabilities block
        prob_offset = batch_id * (head_count * seq_len * seq_len) + \
                     head_id * (seq_len * seq_len) + \
                     (seq_id + offs_m[:, None]) * seq_len + \
                     (start_n + offs_n[None, :])
        p = tl.load(Prob_ptr + prob_offset, mask=(seq_id + offs_m[:, None] < seq_len) & \
                                               (start_n + offs_n[None, :] < seq_len))

        # Compute values block
        v_offset = batch_id * (head_count * seq_len * head_dim) + \
                  head_id * (seq_len * head_dim) + \
                  (start_n + offs_n[:, None]) * head_dim + \
                  offs_d[None, :]
        v = tl.load(V_ptr + v_offset, mask=(start_n + offs_n[:, None] < seq_len))

        # Compute matrix multiplication
        acc += tl.dot(p, v)

    # Write output
    out_offset = batch_id * (head_count * seq_len * head_dim) + \
                 head_id * (seq_len * head_dim) + \
                 (seq_id + offs_m[:, None]) * head_dim + \
                 offs_d[None, :]
    
    tl.store(Out_ptr + out_offset, acc, mask=(seq_id + offs_m[:, None] < seq_len))

def token_att_fwd2(prob, v):
    """
    Wrapper function for the token attention forward kernel
    Args:
        prob: attention probability tensor (batch_size, head_count, seq_len, seq_len)
        v: values tensor (batch_size, head_count, seq_len, head_dim)
    Returns:
        out: output tensor (batch_size, head_count, seq_len, head_dim)
    """
    batch_size, head_count, seq_len, _ = prob.shape
    _, _, _, head_dim = v.shape
    
    # Allocate output
    out = torch.empty_like(v)
    
    # Configure block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = 32

    # Launch kernel
    grid = (batch_size * head_count * triton.cdiv(seq_len, BLOCK_M), )
    _fwd_kernel_token_att2[grid](
        prob, v, out,
        batch_size, head_count, seq_len, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
    
    return out
