import torch
import triton
import triton.language as tl
from flashnn.kernel_backend import get_kernel_meta

@torch.inference_mode()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, max_len_in_batch, output, output_logexpsum):
    """
    Args:
        mid_out: [batch, head, seq_block_num, head_dim]
        mid_out_logexpsum: [batch, head, seq_block_num]
        B_Seqlen: [batch]
        max_len_in_batch: int
        output: [batch, head, head_dim]
        output_logexpsum: [batch, head]
    """
    batch, head, seq_block_num, head_dim = mid_out.shape
    assert mid_out.shape == (batch, head, seq_block_num, head_dim)
    assert mid_out_logexpsum.shape == (batch, head, seq_block_num)
    assert B_Seqlen.numel() == batch
    assert output.shape == (batch, head, head_dim)
    assert output_logexpsum.shape == (batch, head)

    BLOCK_SEQ = triton.next_power_of_2(max_len_in_batch)
    BLOCK_DMODEL = head_dim
    if BLOCK_SEQ < 16:
        BLOCK_SEQ = 16

    kernel_meta = get_kernel_meta(mid_out)
    _fwd_kernel_flash_decode_stage2[(batch, head)](
        B_Seqlen,
        mid_out,  # [batch, head, seq_block_num, head_dim]
        mid_out_logexpsum,  # [batch, head, seq_block_num]
        output,  # [batch, head, head_dim]
        output_logexpsum,  # [batch, head]
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        output_logexpsum.stride(0), output_logexpsum.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=1,
        num_stages=2,
        **kernel_meta
    )
    return
