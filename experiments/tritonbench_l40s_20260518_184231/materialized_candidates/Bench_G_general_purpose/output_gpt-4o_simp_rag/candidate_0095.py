import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,  # Tensor containing sequence lengths for each batch
    Mid_O,  # Intermediate output tensor [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # Logarithm of the exponential sum for each block [batch, head, seq_block_num]
    O,  # Output tensor [batch, head, head_dim]
    out_logexpsum,  # Output log exponential sum [batch, head]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,  # Strides for Mid_O
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,  # Strides for Mid_O_LogExpSum
    stride_obs, stride_oh, stride_od,  # Strides for O
    stride_out_logexpsum_b, stride_out_logexpsum_h,  # Strides for out_logexpsum
    BLOCK_SEQ: tl.constexpr,  # Block size for sequences
    BLOCK_DMODEL: tl.constexpr  # Block size for head dimensions
):
    # Identify the current batch and head from the program IDs
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # Create an offset array for the head dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load the sequence length for the current batch
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # Calculate the number of sequence blocks
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    # Initialize accumulators and logic variables
    sum_exp = 0.0
    max_logic = float("-1e20")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Calculate initial offsets for Mid_O and Mid_O_LogExpSum
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    # Loop over each sequence block
    for block_seq_n in range(0, block_n_size, 1):
        # Load the value and logic sum for the current block
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n * stride_mid_o_es)
        
        # Update max logic and calculate scaling factors
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        
        # Update accumulators
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    # Store the results if there are sequence blocks
    if block_n_size > 0:
        tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
        tl.store(out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h, max_logic + tl.log(sum_exp))
