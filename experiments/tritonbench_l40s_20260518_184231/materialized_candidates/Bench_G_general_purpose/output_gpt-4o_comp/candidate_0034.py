import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def _fwd_kernel_token_att2(Prob, V, Out, Req_to_tokens, cur_batch, cur_head, seq_len, stride_prob_batch, stride_prob_head, stride_prob_m, stride_prob_n, stride_v_batch, stride_v_head, stride_v_m, stride_v_n, stride_out_batch, stride_out_head, stride_out_m, stride_out_n, BLOCK_N: tl.constexpr):
    # Compute the starting position for the block
    pid = tl.program_id(axis=0)
    n_start = pid * BLOCK_N

    # Check if we are within the sequence length
    mask = n_start + tl.arange(0, BLOCK_N) < seq_len

    # Load slices of the Prob and V tensors
    p_ptrs = Prob + cur_batch * stride_prob_batch + cur_head * stride_prob_head + tl.arange(0, BLOCK_N)[:, None] * stride_prob_m + n_start * stride_prob_n
    v_ptrs = V + cur_batch * stride_v_batch + cur_head * stride_v_head + n_start * stride_v_m + tl.arange(0, BLOCK_N) * stride_v_n

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # Load values with masking
    p_value = tl.load(p_ptrs, mask=mask, other=0.0)
    v_value = tl.load(v_ptrs, mask=mask, other=0.0)

    # Compute the attention by multiplying and accumulating
    acc += p_value * v_value

    # Cast accumulator to output data type and store in output tensor
    out_ptrs = Out + cur_batch * stride_out_batch + cur_head * stride_out_head + tl.arange(0, BLOCK_N) * stride_out_m + n_start * stride_out_n
    tl.store(out_ptrs, acc, mask=mask)

# Python wrapper function
@torch.no_grad()
def token_att_fwd2(Prob, V, Out, Req_to_tokens, seq_len, batch_size, num_heads, BLOCK=128):
    # Compute grid dimensions
    grid = (triton.cdiv(seq_len, BLOCK), batch_size, num_heads)

    # Determine strides for each tensor
    stride_prob_batch, stride_prob_head, stride_prob_m, stride_prob_n = Prob.stride()
    stride_v_batch, stride_v_head, stride_v_m, stride_v_n = V.stride()
    stride_out_batch, stride_out_head, stride_out_m, stride_out_n = Out.stride()

    # Launch the Triton kernel
    _fwd_kernel_token_att2[grid](
        Prob, V, Out, Req_to_tokens,
        seq_len=seq_len,
        stride_prob_batch=stride_prob_batch, stride_prob_head=stride_prob_head, stride_prob_m=stride_prob_m, stride_prob_n=stride_prob_n,
        stride_v_batch=stride_v_batch, stride_v_head=stride_v_head, stride_v_m=stride_v_m, stride_v_n=stride_v_n,
        stride_out_batch=stride_out_batch, stride_out_head=stride_out_head, stride_out_m=stride_out_m, stride_out_n=stride_out_n,
        BLOCK_N=BLOCK
    )

# Example usage
# Prob, V, and Out are torch tensors representing probability matrix, value tensor, and output tensor respectively
# Req_to_tokens is an auxiliary input mapping requests to tokens
# seq_len, batch_size, num_heads are integers defining sequence length, batch size, and number of heads
# token_att_fwd2(Prob, V, Out, Req_to_tokens, seq_len, batch_size, num_heads)
