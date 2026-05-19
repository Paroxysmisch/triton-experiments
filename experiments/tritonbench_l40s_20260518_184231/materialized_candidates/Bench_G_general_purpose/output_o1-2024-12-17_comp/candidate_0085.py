import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    Mid_O_ptr, Mid_O_LogExpSum_ptr, B_Seqlen_ptr, O_ptr,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_logexp_b, stride_logexp_h, stride_logexp_s,
    stride_out_b, stride_out_h, stride_out_d,
    head_dim, seq_block_count,
    BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    # Load sequence length for current batch (optional usage, if needed).
    seq_len = tl.load(B_Seqlen_ptr + b_idx)

    # Index range for the vectorized dimension.
    d_offset = tl.arange(0, BLOCK_DMODEL)
    partial_val = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    partial_logexp = -float('inf')

    # Loop over sequence blocks.
    for block_id in range(1024):
        block_valid = block_id < seq_block_count
        logexp_ptr = Mid_O_LogExpSum_ptr + b_idx*stride_logexp_b + h_idx*stride_logexp_h + block_id*stride_logexp_s
        logexp = tl.where(block_valid, tl.load(logexp_ptr), -float('inf'))
        logexp_max = tl.maximum(partial_logexp, logexp)

        scale_old = tl.exp(partial_logexp - logexp_max)
        scale_new = tl.exp(logexp - logexp_max)
        partial_val *= scale_old

        mid_o_offset = (b_idx*stride_mid_ob + h_idx*stride_mid_oh + block_id*stride_mid_os)
        mid_o_ptr = Mid_O_ptr + mid_o_offset + d_offset
        mid_o_val = tl.where((d_offset < head_dim) & block_valid, tl.load(mid_o_ptr), 0.0)
        partial_val += mid_o_val * scale_new

        partial_logexp = logexp_max + tl.log(tl.exp(partial_logexp - logexp_max) + tl.exp(logexp - logexp_max))

        if not block_valid:
            break

    inv_denom = tl.exp(-partial_logexp)
    partial_val = partial_val * inv_denom

    out_offset = b_idx*stride_out_b + h_idx*stride_out_h + d_offset
    mask = d_offset < head_dim
    tl.store(O_ptr + out_offset, tl.where(mask, partial_val, 0.0))

def flash_decode_stage2(Mid_O, Mid_O_LogExpSum, B_Seqlen, O, BLOCK_SEQ=128, BLOCK_DMODEL=128):
    """
    Mid_O:             [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum:   [batch, head, seq_block_num]
    B_Seqlen:          [batch]
    O:                 [batch, head, head_dim]
    """
    batch, head, seq_block_num, head_dim = Mid_O.shape

    assert head_dim % BLOCK_DMODEL == 0 or head_dim <= BLOCK_DMODEL, \
        "head_dim must be compatible with BLOCK_DMODEL"

    grid = (batch, head)
    seq_block_count = seq_block_num

    _fwd_kernel_flash_decode_stage2[grid](
        Mid_O, Mid_O_LogExpSum, B_Seqlen, O,
        Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1), Mid_O_LogExpSum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        head_dim, seq_block_count,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
