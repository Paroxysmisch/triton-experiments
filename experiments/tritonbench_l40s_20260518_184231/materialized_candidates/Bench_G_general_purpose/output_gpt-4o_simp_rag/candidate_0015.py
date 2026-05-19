import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen,
    sm_scale, BLOCK: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_seq_id = tl.program_id(2)

    start_loc = B_Start_Loc[batch_id]
    seqlen = B_Seqlen[batch_id]

    q_ptr = Q + (batch_id * seqlen + q_seq_id) * BLOCK
    k_ptr = K + batch_id * seqlen * BLOCK
    v_ptr = V + batch_id * seqlen * BLOCK
    out_ptr = Out + (batch_id * seqlen + q_seq_id) * BLOCK

    q = tl.load(q_ptr)
    accum = tl.zeros([BLOCK], dtype=tl.float32)
    max_score = tl.full([BLOCK], float('-inf'), dtype=tl.float32)

    for k_seq_id in range(0, seqlen, BLOCK):
        k = tl.load(k_ptr + k_seq_id * BLOCK)
        v = tl.load(v_ptr + k_seq_id * BLOCK)

        qk = tl.dot(q, k) * sm_scale
        max_score = tl.maximum(max_score, qk)
        score = tl.math.exp(qk - max_score)
        accum += score * v

    out = accum / tl.math.exp(max_score)
    tl.store(out_ptr, out)

import torch

def context_attention_fwd(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, BLOCK=64):
    # Assuming Q, K, V are torch tensors of shape [Batch, Heads, SeqLen, D]
    batch_size, num_heads, seq_len, d_model = Q.shape

    # Output tensor
    Out = torch.empty_like(Q)

    # Launch kernel
    grid = (batch_size, num_heads, seq_len // BLOCK)
    num_warps = 4  # Adjust based on your GPU architecture and problem size

    _fwd_kernel[grid](Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale, BLOCK, num_warps=num_warps)

    return Out
