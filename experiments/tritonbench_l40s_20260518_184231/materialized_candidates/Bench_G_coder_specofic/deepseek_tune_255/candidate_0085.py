import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    off_m = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    off_o = cur_batch * stride_o_b + cur_head * stride_o_h

    sum_exp = 0.0
    max_logic = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_s in range(0, cur_batch_seq_len, BLOCK_SEQ):
        start_s = tl.multiple_of(start_s, BLOCK_SEQ)
        end_s = tl.minimum(start_s + BLOCK_SEQ, cur_batch_seq_len)
        seq_len = end_s - start_s

        offs_s = start_s + tl.arange(0, seq_len)
        mask_s = offs_s < cur_batch_seq_len

        new_logic = tl.load(
            Mid_O_LogExpSum + off_m + offs_s, mask=mask_s, other=0.0
        )
        old_logic = tl.where(max_logic > new_logic, max_logic, new_logic)
        max_logic = old_logic

        new_exp = tl.exp(new_logic - old_logic)
        acc *= new_exp

        off_n = off_m + offs_s * stride_mid_os + offs_d * stride_mid_od
        mid_val = tl.load(Mid_O + off_n, mask=mask_s[:, None] & (offs_d < BLOCK_DMODEL), other=0.0)
        acc += mid_val

        sum_exp = sum_exp * new_exp + tl.sum(mid_val, axis=0)

    acc = acc / sum_exp
    tl.store(O + off_o + offs_d, acc, mask=offs_d < BLOCK_DMODEL)
    return

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, logic_b_seq, logic_b_start_s, logic_b_end_s, logic_b_seq_lens, logic_b_blk_lens, logic_b_blk_start_s, logic_b_blk_end_s, logic_b_blk_start_logic, logic_b_blk_end_logic, logic_b_blk_start_exp, logic_b_blk_end_exp, logic_b_blk_logic_diff_scale, logic_b_blk_exp_scale, logic_b_blk_sum_scale, logic_b_blk_diff_scale, logic_b_blk_diff2_scale, logic_b_blk_sum_exp, logic_b_blk_num_seqs, logic_b_blk_num_seqs_logic0, logic_b_blk_num_seqs_logicnon0, logic_b_blk_logic_diff_scale_logic0, logic_b_blk_logic_diff_scale_logicnon0, logic_b_blk_exp_scale_logic0, logic_b_blk_exp_scale_logicnon0, logic_b_blk_sum_scale_logic0, logic_b_blk_sum_scale_logicnon0, logic_b_blk_diff_scale_logic0, logic_b_blk_diff_scale_logicnon0, logic_b_blk_diff2_scale_logic0, logic_b_blk_diff2_scale_logicnon0, logic_b_blk_num_seqs_scale, logic_b_blk_num_seqs_scale_logic0, logic_b_blk_num_seqs_scale_logicnon0, logic_b_blk_num_seqs_logic0_scale, logic_b_blk_num_seqs_logic0_scale_logicnon0, logic_b_blk_num_seqs_logicnon0_scale, logic_b_blk_num_seqs_logicnon0_scale_logic0, BLOCK_SEQ, Lk):
    B, H, _, D = mid_out.shape
    assert Lk in [16, 32, 64, 128, 256, 512]
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    num_warps = 4

    grid = (B, H)
    o_b_seq = torch.empty(B, H, D, device=mid_out.device, dtype=mid_out.dtype)

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        o_b_seq,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        o_b_seq.stride(0),
        o_b_seq.stride(1),
        o_b_seq.stride(2),
        BLOCK_SEQ,
        BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )
    return o_b_seq
