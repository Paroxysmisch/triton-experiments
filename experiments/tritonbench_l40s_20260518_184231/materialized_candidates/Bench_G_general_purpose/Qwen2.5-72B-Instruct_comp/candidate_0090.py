import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen,
    stride_logics_b, stride_logics_h, stride_logics_n,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)  # batch
    hid = tl.program_id(1)  # head

    # Load metadata
    b_loc = tl.load(B_Loc + pid)
    b_start_loc = tl.load(B_Start_Loc + pid)
    b_seqlen = tl.load(B_Seqlen + pid)

    # Initialize pointers
    logits_ptr = Logics + pid * stride_logics_b + hid * stride_logics_h
    v_ptr = V + pid * stride_v_b + hid * stride_v_h
    out_ptr = Out + pid * stride_out_b + hid * stride_out_h

    # Initialize accumulators
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    e_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    e_sum = tl.zeros((1,), dtype=tl.float32)

    # Iterate over blocks of the sequence
    for n in range(0, b_seqlen, BLOCK_N):
        # Load logits and values
        logits = tl.load(logits_ptr + n * stride_logics_n, mask=n + tl.arange(0, BLOCK_N) < b_seqlen, other=-float('inf'))
        v = tl.load(v_ptr + n * stride_v_n, mask=n + tl.arange(0, BLOCK_N) < b_seqlen, other=0.0)

        # Compute max exponential
        e = tl.exp(logits - e_max)
        e_max = tl.maximum(e_max, tl.max(logits, axis=0))
        e_sum += tl.sum(e, axis=0)

        # Compute probabilities
        p = e / e_sum

        # Compute weighted sum
        acc += tl.dot(p, v, allow_tf32=True)

    # Normalize and store the result
    acc /= e_sum
    tl.store(out_ptr, acc, mask=tl.arange(0, BLOCK_DMODEL) < BLOCK_DMODEL)

import torch

def token_softmax_reducev_fwd(logics, v, out, b_loc, b_start_loc, b_seqlen, block_dmodel, block_n, num_warps=4, num_stages=2):
    # Get grid and block dimensions
    grid = (logics.shape[0], logics.shape[1])  # (batch, head)
    block = (block_n,)

    # Get strides
    stride_logics_b = logics.stride(0)
    stride_logics_h = logics.stride(1)
    stride_logics_n = logics.stride(2)
    stride_v_b = v.stride(0)
    stride_v_h = v.stride(1)
    stride_v_n = v.stride(2)
    stride_v_d = v.stride(3)
    stride_out_b = out.stride(0)
    stride_out_h = out.stride(1)
    stride_out_d = out.stride(2)

    # Launch the kernel
    _fwd_kernel[grid, block](
        logics, v, out, b_loc, b_start_loc, b_seqlen,
        stride_logics_b, stride_logics_h, stride_logics_n,
        stride_v_b, stride_v_h, stride_v_n, stride_v_d,
        stride_out_b, stride_out_h, stride_out_d,
        block_dmodel, block_n,
        num_warps=num_warps, num_stages=num_stages
    )
