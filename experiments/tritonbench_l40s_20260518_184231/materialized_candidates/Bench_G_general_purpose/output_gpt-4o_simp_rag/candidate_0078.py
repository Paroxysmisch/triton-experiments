import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    out_logexpsum,  # [batch, head]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_out_logexpsum_b, stride_out_logexpsum_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr):
    
    # Get the current batch and head based on program ID
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # Determine the offsets for the head dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load the sequence length for the current batch
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # Calculate the number of blocks in the sequence
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Initialize variables for accumulation and max logic
    sum_exp = 0.0
    max_logic = float("-1e20")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Calculate offsets for accessing Mid_O and Mid_O_LogExpSum
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    # Loop over each block in the sequence
    for block_seq_n in range(0, block_n_size, 1):
        # Load the current block's data and log-exp sum
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)

        # Update the maximum logic and calculate scaling factors
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        exp_logic = tl.exp(tlogic - new_max_logic)

        # Update the accumulator and sum of exponentials
        acc *= old_scale
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    # Store the normalized results in the output tensors
    if block_n_size > 0:
        tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
        tl.store(out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h, max_logic + tl.log(sum_exp))
