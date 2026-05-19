import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen_p, Mid_O_p, Mid_O_LogExpSum_p, Out_p,
    stride_B_Seqlen, stride_Mid_O, stride_Mid_O_LogExpSum, stride_Out,
    BLOCK_SEQ, BLOCK_DMODEL,
    grid
):
    # Identify the current batch and head using `tl.program_id`.
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    # Initialize accumulators
    sum_exp = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
    max_logic = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SEQ, BLOCK_DMODEL), dtype=tl.float32)

    # Load current sequence length and calculate number of sequence blocks
    B_Seqlen = tl.load(B_Seqlen_p + batch_id * stride_B_Seqlen)
    block_n_size = (B_Seqlen + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Iterate over each block
    for block_id in range(block_n_size):
        # Load values and logic sums
        tv = tl.load(Mid_O_p + head_id * stride_Mid_O + block_id * BLOCK_SEQ * stride_Mid_O)
        tlogic = tl.load(Mid_O_LogExpSum_p + head_id * stride_Mid_O_LogExpSum + block_id * BLOCK_SEQ * stride_Mid_O_LogExpSum)

        # Compute maximum logic value across blocks and scale previous accumulations
        max_logic = tl.max(max_logic, tlogic)
        acc = acc * tl.exp(tlogic - max_logic)

        # Update accumulators by computing exponential of adjusted logic values and scaling/accumulating
        acc = acc + tv * tl.exp(tlogic - max_logic)

    # Store final normalized result into Out, scaling accumulated values by the sum of exponentials
    Out = acc / tl.sum(acc, axis=0)
    tl.store(Out_p + head_id * stride_Out, Out)
