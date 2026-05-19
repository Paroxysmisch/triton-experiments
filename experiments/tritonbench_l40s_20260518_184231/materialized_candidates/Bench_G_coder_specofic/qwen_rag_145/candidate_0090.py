import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, BLOCK_N,
                 BLOCK_DMODEL: tl.constexpr):
    program_id = tl.program_id(0)
    batch_idx = program_id // tl.num_programs(1)
    head_idx = program_id % tl.num_programs(1)
    batch_start = B_Start_Loc[batch_idx]
    batch_seq_len = B_Seqlen[batch_idx]

    # Grid-stride loop to process sequence
    for seq_idx in range(tl.program_id(1), batch_seq_len, BLOCK_N):
        # Load index and value
        index = tl.load(Logics + batch_start + seq_idx)
        v = tl.load(V + batch_start + index * BLOCK_DMODEL)

        # Calculate exponential and find maximum
        exp_v = tl.exp(v)
        e_max = tl.max(exp_v)

        # Apply max-shifted exponentiation and calculate probability
        shifted = exp_v - e_max
        p = shifted / tl.sum(shifted)

        # Perform the accumulation
        acc = tl.dot(p, v)

        # Store the result
        tl.store(Out + batch_start + seq_idx, acc)
