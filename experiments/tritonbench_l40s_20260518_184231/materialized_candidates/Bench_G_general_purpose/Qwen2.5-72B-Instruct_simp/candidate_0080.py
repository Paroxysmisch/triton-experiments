import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob,  # Probabilities tensor
    V,     # Values tensor
    Out,   # Output tensor
    stride_prob_b, stride_prob_h, stride_prob_s, stride_prob_t,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr
):
    # Get the batch and head indices
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # Pointers to the start of the batch and head
    prob_ptr = Prob + bid * stride_prob_b + hid * stride_prob_h
    v_ptr = V + bid * stride_v_b + hid * stride_v_h
    out_ptr = Out + bid * stride_out_b + hid * stride_out_h

    # Iterate over the sequence in blocks
    for start_t in range(0, stride_prob_s, BLOCK_SEQ):
        # Load the probabilities for the current block
        prob = tl.load(prob_ptr + start_t * stride_prob_s, mask=start_t + tl.arange(0, BLOCK_SEQ) < stride_prob_s, other=0.0)

        # Initialize the output block
        out = tl.zeros((BLOCK_SEQ, BLOCK_DMODEL), dtype=tl.float32)

        # Iterate over the sequence to compute the weighted sum
        for s in range(stride_prob_t):
            # Load the values for the current sequence position
            v = tl.load(v_ptr + s * stride_v_s + start_t * stride_v_d, mask=start_t + tl.arange(0, BLOCK_SEQ) < stride_prob_s, other=0.0)

            # Compute the weighted sum
            out += prob[:, None] * v

        # Store the result in the output tensor
        tl.store(out_ptr + start_t * stride_out_d, out, mask=start_t + tl.arange(0, BLOCK_SEQ) < stride_prob_s)

        # Move to the next block
        prob_ptr += BLOCK_SEQ * stride_prob_s
        v_ptr += BLOCK_SEQ * stride_v_d
        out_ptr += BLOCK_SEQ * stride_out_d

import torch

def token_att_fwd2(Prob, V, Out, B, H, S, D, BLOCK_SEQ=128, BLOCK_DMODEL=64):
    # Get the strides for the input tensors
    stride_prob_b = Prob.stride(0)
    stride_prob_h = Prob.stride(1)
    stride_prob_s = Prob.stride(2)
    stride_prob_t = Prob.stride(3)

    stride_v_b = V.stride(0)
    stride_v_h = V.stride(1)
    stride_v_s = V.stride(2)
    stride_v_d = V.stride(3)

    stride_out_b = Out.stride(0)
    stride_out_h = Out.stride(1)
    stride_out_s = Out.stride(2)
    stride_out_d = Out.stride(3)

    # Launch the kernel
    grid = (B, H)
    _fwd_kernel_token_att2[grid](
        Prob, V, Out,
        stride_prob_b, stride_prob_h, stride_prob_s, stride_prob_t,
        stride_v_b, stride_v_h, stride_v_s, stride_v_d,
        stride_out_b, stride_out_h, stride_out_s, stride_out_d,
        BLOCK_DMODEL, BLOCK_SEQ
    )

# Example usage
B = 2  # Batch size
H = 4  # Number of heads
S = 128  # Sequence length
D = 64  # Model dimension

Prob = torch.randn((B, H, S, S), device='cuda')
V = torch.randn((B, H, S, D), device='cuda')
Out = torch.zeros((B, H, S, D), device='cuda')

token_att_fwd2(Prob, V, Out, B, H, S, D)
