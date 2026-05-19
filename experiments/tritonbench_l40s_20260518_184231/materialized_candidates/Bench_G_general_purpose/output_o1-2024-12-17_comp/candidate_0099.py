import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen_ptr, 
    Mid_O_ptr, 
    Mid_O_LogExpSum_ptr, 
    Out_ptr,
    stride_seqlen,
    stride_mid_o_b,
    stride_mid_o_h,
    stride_mid_o_s,
    stride_mid_o_d,
    stride_logexp_b,
    stride_logexp_h,
    stride_logexp_s,
    stride_logexp_d,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load sequence length for this batch
    # B_Seqlen shape: [batch]
    seqlen = tl.load(B_Seqlen_ptr + batch_id * stride_seqlen)
    block_n_size = (seqlen + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Prepare accumulators
    d_offset = tl.arange(0, BLOCK_DMODEL)
    max_logic = tl.full([BLOCK_DMODEL], -1e9, dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Process each sequence block
    for sq_block_idx in range(0, block_n_size):
        seq_offset = sq_block_idx * BLOCK_SEQ

        # Load logarithmic sums
        base_logexp_ptr = (
            Mid_O_LogExpSum_ptr
            + batch_id * stride_logexp_b
            + head_id * stride_logexp_h
            + seq_offset * stride_logexp_s
        )
        tlogic = tl.load(
            base_logexp_ptr + d_offset * stride_logexp_d,
            mask=(d_offset < BLOCK_DMODEL),
            other=-1e9
        )
        local_max = tl.max(tlogic, 0)

        # Merge new logic maxima with previous accumulations
        max_logic_new = tl.maximum(max_logic, local_max)
        scale = tl.exp(max_logic - max_logic_new)
        acc *= scale
        sum_exp *= scale

        # Accumulate exponential values
        exp_vals = tl.exp(tlogic - max_logic_new)
        sum_exp += exp_vals

        # Load intermediate values
        base_mid_o_ptr = (
            Mid_O_ptr
            + batch_id * stride_mid_o_b
            + head_id * stride_mid_o_h
            + seq_offset * stride_mid_o_s
        )
        tv = tl.load(
            base_mid_o_ptr + d_offset * stride_mid_o_d,
            mask=(d_offset < BLOCK_DMODEL),
            other=0.0
        )
        acc += exp_vals * tv
        max_logic = max_logic_new

    # Normalize and store the result
    out_val = acc / sum_exp
    base_out_ptr = Out_ptr + batch_id * stride_out_b + head_id * stride_out_h
    tl.store(
        base_out_ptr + d_offset * stride_out_d,
        out_val,
        mask=(d_offset < BLOCK_DMODEL)
    )


def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, BLOCK_SEQ=128, BLOCK_DMODEL=128):
    """
    B_Seqlen shape:          [batch]
    Mid_O shape:             [batch, head, seq, dmodel]
    Mid_O_LogExpSum shape:   [batch, head, seq, dmodel]
    Out shape:               [batch, head, dmodel]
    """
    batch = B_Seqlen.shape[0]
    head_num = Mid_O.shape[1]

    # Strides
    stride_seqlen = B_Seqlen.stride(0)
    stride_mid_o_b = Mid_O.stride(0)
    stride_mid_o_h = Mid_O.stride(1)
    stride_mid_o_s = Mid_O.stride(2)
    stride_mid_o_d = Mid_O.stride(3)
    stride_logexp_b = Mid_O_LogExpSum.stride(0)
    stride_logexp_h = Mid_O_LogExpSum.stride(1)
    stride_logexp_s = Mid_O_LogExpSum.stride(2)
    stride_logexp_d = Mid_O_LogExpSum.stride(3)
    stride_out_b = Out.stride(0)
    stride_out_h = Out.stride(1)
    stride_out_s = 0  # Not used when Out is of shape [batch, head, dmodel]
    stride_out_d = Out.stride(2)

    grid = (batch, head_num)
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, 
        Mid_O, 
        Mid_O_LogExpSum, 
        Out,
        stride_seqlen,
        stride_mid_o_b,
        stride_mid_o_h,
        stride_mid_o_s,
        stride_mid_o_d,
        stride_logexp_b,
        stride_logexp_h,
        stride_logexp_s,
        stride_logexp_d,
        stride_out_b,
        stride_out_h,
        stride_out_s,
        stride_out_d,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
