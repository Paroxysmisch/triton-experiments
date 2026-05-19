import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_o_b, stride_o_h, stride_o_d,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + (block_n_size - 1)
    for block_seq_n in range(0, block_n_size, 1):
        # [BLOCK_DMODEL]
        v = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        # [1]
        logic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        # if-else logic is not supported, change to where
        max_logic_new = tl.maximum(max_logic, logic)
        # [BLOCK_DMODEL]
        exp_logic = tl.exp(logic - max_logic_new)
        # [BLOCK_DMODEL]
        v_new = v * exp_logic
        # [BLOCK_DMODEL]
        acc *= tl.exp(max_logic - max_logic_new)
        acc += v_new
        sum_exp = sum_exp * tl.exp(max_logic - max_logic_new) + exp_logic
        max_logic = max_logic_new

    tl.store(O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d, acc / sum_exp)
    return

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O):
    """
    Args:
        mid_out: [batch, head, seq_block_num, head_dim]
        mid_out_logexpsum: [batch, head, seq_block_num]
        B_Seqlen: [batch]
        O: [batch, head, head_dim]
    """
    batch, head, seq_block_num, d_model = mid_out.shape
    assert d_model == O.shape[-1]
    assert B_Seqlen.shape[0] == batch

    grid = (batch, head)
    BLOCK_SEQ = 16
    BLOCK_DMODEL = triton.next_power_of_2(d_model)
    num_warps = 1

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        O,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )
    return
