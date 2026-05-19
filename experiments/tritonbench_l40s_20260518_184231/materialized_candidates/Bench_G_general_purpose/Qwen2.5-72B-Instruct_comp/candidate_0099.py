import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,  # Pointer to sequence lengths per batch
    Mid_O,     # Pointer to intermediate outputs
    Mid_O_LogExpSum,  # Pointer to log-exp sum values
    Out,       # Pointer to final output
    stride_b,  # Stride for B_Seqlen
    stride_mid_o_b,  # Stride for Mid_O in batch dimension
    stride_mid_o_h,  # Stride for Mid_O in head dimension
    stride_mid_o_s,  # Stride for Mid_O in sequence dimension
    stride_mid_o_d,  # Stride for Mid_O in dimension alignment
    stride_mid_o_logexpsum_b,  # Stride for Mid_O_LogExpSum in batch dimension
    stride_mid_o_logexpsum_h,  # Stride for Mid_O_LogExpSum in head dimension
    stride_out_b,  # Stride for Out in batch dimension
    stride_out_h,  # Stride for Out in head dimension
    stride_out_s,  # Stride for Out in sequence dimension
    stride_out_d,  # Stride for Out in dimension alignment
    batch,  # Batch size
    head_num,  # Number of heads
    seqlen,  # Sequence length
    dmodel,  # Dimension alignment
    BLOCK_SEQ: tl.constexpr,  # Sequence block size
    BLOCK_DMODEL: tl.constexpr  # Dimension alignment block size
):
    # Identify the current batch and head
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # Initialize accumulators
    sum_exp = tl.zeros([], dtype=tl.float32)
    max_logic = tl.full([], -float('inf'), dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Load the current sequence length
    seqlen_b = tl.load(B_Seqlen + bid * stride_b)
    block_n_size = (seqlen_b + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Iterate over each block
    for block_n in range(block_n_size):
        # Calculate the start and end indices for the current block
        start_n = block_n * BLOCK_SEQ
        end_n = tl.minimum(start_n + BLOCK_SEQ, seqlen_b)

        # Load values from Mid_O and logic sums from Mid_O_LogExpSum
        tv = tl.load(Mid_O + bid * stride_mid_o_b + hid * stride_mid_o_h + start_n * stride_mid_o_s + tl.arange(0, BLOCK_DMODEL) * stride_mid_o_d, mask=start_n + tl.arange(0, BLOCK_SEQ) < seqlen_b, other=0.0)
        tlogic = tl.load(Mid_O_LogExpSum + bid * stride_mid_o_logexpsum_b + hid * stride_mid_o_logexpsum_h + start_n * stride_mid_o_s, mask=start_n + tl.arange(0, BLOCK_SEQ) < seqlen_b, other=0.0)

        # Compute the maximum logic value across blocks
        max_logic = tl.maximum(max_logic, tlogic)

        # Scale previous accumulations
        sum_exp = sum_exp * tl.exp(max_logic - tlogic)
        acc = acc * tl.exp(max_logic - tlogic)

        # Update the accumulators
        exp_val = tl.exp(tv - tlogic)
        sum_exp += tl.sum(exp_val, axis=0)
        acc += exp_val

    # Store the final normalized result into Out
    out_ptr = Out + bid * stride_out_b + hid * stride_out_h + tl.arange(0, BLOCK_DMODEL) * stride_out_d
    tl.store(out_ptr, acc / sum_exp, mask=tl.arange(0, BLOCK_DMODEL) < dmodel)

import torch
import triton
import triton.language as tl

def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, batch, head_num, seqlen, dmodel, BLOCK_SEQ=128, BLOCK_DMODEL=64):
    # Determine grid and block dimensions
    grid = (batch, head_num)

    # Launch the kernel
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,  # Pointer to sequence lengths per batch
        Mid_O,     # Pointer to intermediate outputs
        Mid_O_LogExpSum,  # Pointer to log-exp sum values
        Out,       # Pointer to final output
        B_Seqlen.stride(0),  # Stride for B_Seqlen
        Mid_O.stride(0),  # Stride for Mid_O in batch dimension
        Mid_O.stride(1),  # Stride for Mid_O in head dimension
        Mid_O.stride(2),  # Stride for Mid_O in sequence dimension
        Mid_O.stride(3),  # Stride for Mid_O in dimension alignment
        Mid_O_LogExpSum.stride(0),  # Stride for Mid_O_LogExpSum in batch dimension
        Mid_O_LogExpSum.stride(1),  # Stride for Mid_O_LogExpSum in head dimension
        Out.stride(0),  # Stride for Out in batch dimension
        Out.stride(1),  # Stride for Out in head dimension
        Out.stride(2),  # Stride for Out in sequence dimension
        Out.stride(3),  # Stride for Out in dimension alignment
        batch,  # Batch size
        head_num,  # Number of heads
        seqlen,  # Sequence length
        dmodel,  # Dimension alignment
        BLOCK_SEQ,  # Sequence block size
        BLOCK_DMODEL  # Dimension alignment block size
    )
