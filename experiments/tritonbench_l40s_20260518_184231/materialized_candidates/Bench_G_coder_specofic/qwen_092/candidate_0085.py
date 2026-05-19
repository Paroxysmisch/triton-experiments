import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    O,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_o_b, stride_o_h, stride_o_d,
    BLOCK_SEQ, BLOCK_DMODEL,
):
    # Get the current block index
    block_id = tl.program_id(0)
    batch_idx = block_id // (BLOCK_SEQ * BLOCK_DMODEL)
    head_idx = (block_id // BLOCK_DMODEL) % BLOCK_SEQ
    seq_block_idx = block_id % BLOCK_DMODEL

    # Compute the starting index for each dimension
    mid_o_base = batch_idx * stride_mid_ob + head_idx * stride_mid_oh + seq_block_idx * stride_mid_os
    mid_o_logexpsum_base = batch_idx * stride_mid_ob + head_idx * stride_mid_oh + seq_block_idx * stride_mid_os
    o_base = batch_idx * stride_o_b + head_idx * stride_o_h

    # Load the log-exp sum value
    log_exp_sum = tl.load(Mid_O_LogExpSum + mid_o_logexpsum_base, mask=seq_block_idx < B_Seqlen[batch_idx])

    # Initialize the accumulation for the output tensor
    o_accum = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Loop over sequence blocks
    for seq_block in range(BLOCK_SEQ):
        mid_o_idx = mid_o_base + seq_block * stride_mid_os
        mid_o = tl.load(Mid_O + mid_o_idx, mask=seq_block < B_Seqlen[batch_idx])

        # Scale the mid_o by the log-exp sum and accumulate
        scaled_mid_o = mid_o * log_exp_sum
        o_accum = o_accum + scaled_mid_o

    # Normalize the accumulation
    o_norm = o_accum / tl.exp(log_exp_sum)

    # Store the result in the output tensor
    o_idx = o_base + seq_block_idx * stride_o_d
    tl.store(O + o_idx, o_norm, mask=seq_block_idx < B_Seqlen[batch_idx])
