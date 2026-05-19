import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 16}, num_warps=8),
        triton.Config({'BLOCK_SEQ': 16}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 16}, num_warps=2),
    ],
    key=['batch_dim', 'head_dim'],
)
@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    out_logexpsum,  # [batch, head]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_out_logexpsum_b, stride_out_logexpsum_h,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = float("-1e20")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    
    if block_n_size > 0:
        # Here we check whether block_n_size is 0 in order to avoid "div by zero" error
        tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
        tl.store(out_logexpsum + cur_batch * stride_out_logexpsum_b + cur_head * stride_out_logexpsum_h, max_logic + tl.log(sum_exp))
    return


def flash_decode_stage2(mid_o: torch.Tensor, mid_o_one: torch.Tensor, out_logexpsum: torch.Tensor, batch_seq_len: torch.Tensor, blank: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    # mid_o: [batch, head, seq_block_num, head_dim]
    # mid_o_one: [batch, head, seq_block_num]
    # batch_seq_len: [batch]
    # out_logexpsum: [batch, head]
    head_num = mid_o.shape[1]
    head_dim = mid_o.shape[-1]
    batch_size = mid_o.shape[0]
    mid_o_LogExpSum = mid_o_one + mid_o.cumsum(dim=2).to(mid_o_one.dtype)
    assert blank >= 0 and blank < head_num, "blank must be in the range [0, head_num)"
    out = torch.empty_like(mid_o[:, :, 0, :])
    grid = (batch_size, head_num)
    _fwd_kernel_flash_decode_stage2[grid](
        batch_seq_len,
        mid_o, mid_o_LogExpSum,
        out, out_logexpsum,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_o_one.stride(0), mid_o_one.stride(1), mid_o_one.stride(2),
        mid_o_LogExpSum.stride(0), mid_o_LogExpSum.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        out_logexpsum.stride(0), out_logexpsum.stride(1),
        BLOCK_DMODEL=head_dim, 
        BLOCK_SEQ=32,
        num_warps=4,
        num_stages=2,
    )
    return out, out_logexpsum
