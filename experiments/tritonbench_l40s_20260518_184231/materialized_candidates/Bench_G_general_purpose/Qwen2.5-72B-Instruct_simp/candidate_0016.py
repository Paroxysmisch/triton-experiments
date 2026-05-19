import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    nheads, seqlen, BLOCK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // nheads
    hid = pid % nheads

    # Pointers for Q, K, V, and Out
    Q += bid * stride_qb + hid * stride_qh
    K += bid * stride_kb + hid * stride_kh
    V += bid * stride_vb + hid * stride_vh
    Out += bid * stride_ob + hid * stride_oh

    # Offsets for the current block
    q_offsets = tl.arange(0, BLOCK)
    k_offsets = tl.arange(0, BLOCK)
    v_offsets = tl.arange(0, BLOCK)

    # Load Q, K, V blocks
    q = tl.load(Q + q_offsets[:, None] * stride_qd)
    k = tl.load(K + k_offsets[None, :] * stride_kd)
    v = tl.load(V + v_offsets[None, :] * stride_vd)

    # Compute the dot-product qk
    qk = tl.dot(q, k, allow_tf32=True) * sm_scale

    # Apply sliding window attention mask
    start_loc = tl.load(B_Start_Loc + bid)
    seqlen = tl.load(B_Seqlen + bid)
    qk = tl.where((q_offsets[:, None] + start_loc) >= (k_offsets[None, :] + start_loc - BLOCK), qk, float('-inf'))

    # Compute softmax and scale
    m = tl.max(qk, 1)
    qk = qk - m[:, None]
    qk = tl.exp(qk)
    qk = qk / tl.sum(qk, 1)[:, None]

    # Compute the output
    out = tl.dot(qk, v, allow_tf32=True)

    # Write the output back to Out
    tl.store(Out + q_offsets[:, None] * stride_od, out)

import triton
import triton.language as tl
import torch

def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, sm_scale, BLOCK=128):
    # Get the shapes
    batch_size, nheads, seqlen, d_head = Q.shape

    # Allocate output tensor
    Out = torch.empty_like(Q)

    # Define grid and block sizes
    grid = (batch_size * nheads,)

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        nheads, seqlen, BLOCK
    )

    return Out
