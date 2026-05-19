triton
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Logics: tl.tensor,  # shape: [B, H, N, D]
    V: tl.tensor,       # shape: [B, H, N, D]
    Out: tl.tensor,     # shape: [B, H, N, D]
    B_Loc: tl.tensor,   # shape: [B, N]
    B_Start_Loc: tl.tensor,  # shape: [B]
    B_Seqlen: tl.tensor,  # shape: [B]
    BLOCK_N: int,       # sequence block size
    BLOCK_DMODEL: int,  # model dimension
    BLOCK_H: int,       # head block size
    grid_program_id: tl.program_id(0),
    grid_program_id_h: tl.program_id(1),
    num_warps: int,
    num_stages: int,
):
    # Get batch and head indices
    b = grid_program_id
    h = grid_program_id_h

    # Get the sequence length for the current batch
    seqlen = B_Seqlen[b]

    # Iterate over blocks of the sequence
    for n in range(0, seqlen, BLOCK_N):
        # Get the start and end indices of the current block
        start_n = n
        end_n = min(n + BLOCK_N, seqlen)

        # Load logits and values for the current block
        logits = tl.load(Logics[b, h, start_n:end_n, :])
        values = tl.load(V[b, h, start_n:end_n, :])

        # Initialize max exponentials, probabilities, and accumulated weighted sum
        e_max = tl.max(logits, axis=1)
        p = tl.exp(logits - e_max[:, None])
        acc = tl.zeros((end_n - start_n, BLOCK_DMODEL), dtype=logits.dtype)

        # Compute the weighted sum for the current block
        for i in range(end_n - start_n):
            acc[i] = tl.sum(p[i] * values[i], axis=0)

        # Normalize the accumulated weighted sum
        sum_p = tl.sum(p, axis=1)
        acc = acc / sum_p[:, None]

        # Store the result in the output tensor
        tl.store(Out[b, h, start_n:end_n, :], acc)
