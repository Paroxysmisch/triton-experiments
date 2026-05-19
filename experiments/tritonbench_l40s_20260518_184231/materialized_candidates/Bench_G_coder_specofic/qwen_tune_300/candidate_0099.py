import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Out,  # [batch, head, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    sum_exp = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    max_logic = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    tv = tl.zeros([BLOCK_SEQ, BLOCK_DMODEL], dtype=tl.float32)
    tlogic = tl.zeros([BLOCK_SEQ], dtype=tl.float32)

    for block_seq_n in range(0, block_n_size, 1):
        # [batch, head, seq_block_num, head_dim] -> [BLOCK_SEQ, BLOCK_DMODEL]
        tv = tl.load(Mid_O + cur_batch * stride_mid_ob + cur_head * stride_mid_oh + (block_seq_n) * stride_mid_os + tl.arange(0, BLOCK_DMODEL), tv, mask=tl.arange(0, BLOCK_DMODEL) < BLOCK_DMODEL, other=0.0)
        # [batch, head, seq_block_num] -> [BLOCK_SEQ]
        tlogic = tl.load(Mid_O_LogExpSum + cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + (block_seq_n), tlogic, mask=True, other=-10000.0)

        # [BLOCK_SEQ] -> [BLOCK_DMODEL]
        cur_max_logic = tl.maximum(tlogic, max_logic)
        # [BLOCK_DMODEL]
        # scale prev acc
        last_acc = tl.exp(max_logic - cur_max_logic) * acc
        # [BLOCK_DMODEL]
        # scale tv by exp(tlogic) and add to acc
        tv = tl.exp(tlogic - cur_max_logic) * tv
        acc = last_acc + tl.sum(tv, 0)

        # [BLOCK_DMODEL]
        max_logic = cur_max_logic

    # [BLOCK_DMODEL]
    # final scale
    acc = acc * tl.exp(max_logic)

    # [BLOCK_DMODEL]
    # save
    tl.store(Out + cur_batch * stride_ob + cur_head * stride_oh + tl.arange(0, BLOCK_DMODEL), acc, mask=tl.arange(0, BLOCK_DMODEL) < BLOCK_DMODEL)

    return

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, seq_len, out):
    # mid_out: [batch, head, seq_block_num, head_dim]
    # mid_out_logexpsum: [batch, head, seq_block_num]
    # seq_len: [batch]
    # out: [batch, head, head_dim]

    batch, head_num, seq_block_num, head_dim = mid_out.shape

    assert seq_block_num == -(-seq_len.max().item() // mid_out.shape[-2])

    mid_out = mid_out.contiguous()
    mid_out_logexpsum = mid_out_logexpsum.contiguous()
    out = out.contiguous()

    grid = (batch, head_num)

    # [batch, head, seq_block_num]
    max_logic = mid_out_logexpsum.max(-1)[0]
    # [batch, head, seq_block_num, head_dim]
    max_logic = max_logic.unsqueeze(-1).expand(batch, head_num, seq_block_num, head_dim)

    # [batch, head, head_dim]
    acc = (mid_out * (mid_out_logexpsum.unsqueeze(-1) - max_logic)).sum(-2)

    # [batch, head, head_dim]
    out = out.view(batch * head_num, head_dim)
    acc = acc.view(batch * head_num, head_dim)

    # [batch, head, head_dim]
    tl.static_print("BLOCK_DMODEL", head_dim)
    _fwd_kernel_flash_decode_stage2[grid](
        seq_len,
        mid_out,
        mid_out_logexpsum,
        out,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logexpsum.stride(0),
        mid_out_logexpsum.stride(1),
        mid_out_logexpsum.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SEQ=4,
        BLOCK_DMODEL=head_dim,
        num_warps=4,
        num_stages=2,
    )

    out = out.view(batch, head_num, head_dim)
    return
