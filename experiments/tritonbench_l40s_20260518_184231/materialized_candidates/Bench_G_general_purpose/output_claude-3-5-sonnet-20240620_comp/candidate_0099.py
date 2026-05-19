import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen, Mid_O, Mid_O_LogExpSum, Out,
    stride_bs, stride_bh, stride_bsd, stride_bo,
    BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    batch = tl.program_id(0)
    head = tl.program_id(1)

    # Initialize accumulators
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Load sequence length for current batch
    seqlen = tl.load(B_Seqlen + batch)
    block_n_size = tl.cdiv(seqlen, BLOCK_SEQ)

    # Iterate over sequence blocks
    for block in range(0, block_n_size):
        # Offsets
        off_m = batch * stride_bs + head * stride_bh + block * BLOCK_SEQ * stride_bsd
        off_l = batch * stride_bs + head * stride_bh + block * BLOCK_SEQ

        # Load values and logic sums
        tv = tl.load(Mid_O + off_m + tl.arange(0, BLOCK_DMODEL))
        tlogic = tl.load(Mid_O_LogExpSum + off_l)

        # Update max logic
        prev_max_logic = max_logic
        max_logic = tl.maximum(max_logic, tlogic)

        # Scale previous accumulations
        sum_exp = sum_exp * tl.exp(prev_max_logic - max_logic)
        acc = acc * tl.exp(prev_max_logic - max_logic)

        # Update accumulators
        curr_exp = tl.exp(tlogic - max_logic)
        sum_exp += curr_exp
        acc += curr_exp * tv

    # Normalize and store result
    acc = acc / sum_exp
    off_o = batch * stride_bo + head * stride_bh
    tl.store(Out + off_o + tl.arange(0, BLOCK_DMODEL), acc)

def flash_decode_stage2(b_seqlen, mid_o, mid_o_logexpsum):
    batch, head_num, max_seqlen, d_model = mid_o.shape
    assert d_model % 32 == 0, "d_model must be a multiple of 32"
    
    # Output tensor
    out = torch.empty_like(mid_o)
    
    # Define block sizes
    BLOCK_SEQ = 16
    BLOCK_DMODEL = d_model
    
    # Launch kernel
    grid = (batch, head_num)
    _fwd_kernel_flash_decode_stage2[grid](
        b_seqlen, mid_o, mid_o_logexpsum, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), out.stride(0),
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=2
    )
    
    return out
