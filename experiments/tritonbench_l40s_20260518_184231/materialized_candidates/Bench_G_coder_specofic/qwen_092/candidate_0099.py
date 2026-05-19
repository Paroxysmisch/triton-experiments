triton
@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen: pointer[int32],
    Mid_O: pointer[float32],
    Mid_O_LogExpSum: pointer[float32],
    Out: pointer[float32],
    B_Seqlen_stride: int32,
    Mid_O_stride: int32,
    Mid_O_LogExpSum_stride: int32,
    Out_stride: int32,
    BLOCK_SEQ: int32,
    BLOCK_DMODEL: int32
):
    # Get the program ID (batch, head)
    batch, head = tl.program_id(0), tl.program_id(1)

    # Initialize accumulators
    sum_exp = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
    max_logic = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SEQ, BLOCK_DMODEL], dtype=tl.float32)

    # Load the current sequence length
    seq_len = B_Seqlen[batch]

    # Calculate the number of sequence blocks
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Iterate over each block
    for b in range(block_n_size):
        # Calculate the start and end indices for the current block
        start_idx = b * BLOCK_SEQ
        end_idx = min(start_idx + BLOCK_SEQ, seq_len)

        # Load values from Mid_O and logic sums from Mid_O_LogExpSum
        tv = tl.load(Mid_O + (batch * Mid_O_stride + head * BLOCK_DMODEL * seq_len + start_idx * BLOCK_DMODEL), mask=end_idx - start_idx)
        tlogic = tl.load(Mid_O_LogExpSum + (batch * Mid_O_LogExpSum_stride + head * seq_len + start_idx), mask=end_idx - start_idx)

        # Compute the maximum logic value across blocks and scale previous accumulations
        max_logic = tl.maximum(max_logic, tlogic)
        acc *= tl.exp(max_logic - tlogic)

        # Update the accumulators by computing the exponential of adjusted logic values and scaling/accumulating
        acc += tv * tl.exp(tlogic - max_logic)

    # Normalize the accumulated values by the sum of exponentials
    sum_exp = tl.sum(acc, axis=1)
    Out[batch * Out_stride + head * seq_len] = sum_exp / seq_len
