import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_SEQ = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen, Mid_O, Mid_O_LogExpSum, Out,
    stride_bs, stride_bm, stride_bl, stride_bo,
    BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Identify the current batch and head
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Initialize accumulators
    sum_exp = tl.zeros([BLOCK_SEQ, BLOCK_DMODEL], dtype=tl.float32)
    max_logic = tl.full([BLOCK_SEQ, BLOCK_DMODEL], -float('inf'), dtype=tl.float32)
    acc = tl.zeros([BLOCK_SEQ, BLOCK_DMODEL], dtype=tl.float32)

    # Load the current sequence length
    seqlen = tl.load(B_Seqlen + batch_id)
    block_n_size = (seqlen + BLOCK_SEQ - 1) // BLOCK_SEQ

    # Iterate over each block
    for block_idx in range(block_n_size):
        # Compute offsets
        offset_m = batch_id * stride_bm + head_id * BLOCK_SEQ + block_idx * BLOCK_SEQ
        offset_l = batch_id * stride_bl + head_id * BLOCK_SEQ + block_idx * BLOCK_SEQ

        # Load values and logic sums
        tv = tl.load(Mid_O + offset_m, mask=block_idx < block_n_size)
        tlogic = tl.load(Mid_O_LogExpSum + offset_l, mask=block_idx < block_n_size)

        # Compute maximum logic value across blocks
        max_logic = tl.maximum(max_logic, tlogic)

        # Scale previous accumulations
        sum_exp *= tl.exp(max_logic - tlogic)
        acc *= tl.exp(max_logic - tlogic)

        # Update accumulators
        exp_logic = tl.exp(tlogic - max_logic)
        sum_exp += exp_logic
        acc += exp_logic * tv

    # Compute final normalized result
    norm_result = acc / sum_exp

    # Store result into Out
    offset_o = batch_id * stride_bo + head_id * BLOCK_SEQ
    tl.store(Out + offset_o, norm_result)

def flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, batch, head_num):
    # Define grid size
    grid = (batch, head_num)

    # Calculate strides
    stride_bs = 1
    stride_bm = BLOCK_SEQ * head_num
    stride_bl = BLOCK_SEQ * head_num
    stride_bo = BLOCK_SEQ * head_num

    # Launch the Triton kernel
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen, Mid_O, Mid_O_LogExpSum, Out,
        stride_bs, stride_bm, stride_bl, stride_bo,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL
    )
