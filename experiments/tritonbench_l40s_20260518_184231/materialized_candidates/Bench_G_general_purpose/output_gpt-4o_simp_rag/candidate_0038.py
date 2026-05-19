import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2(
    P, V, Out,
    B_Start_Loc, B_Seqlen,
    stride_pbs, stride_ph, stride_vbs, stride_vh, stride_obs, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    off_p = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_pbs
        + cur_head * stride_ph
        + offs_n[None, :]
    )
    off_v = offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :]

    p = tl.load(P + off_p, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v = tl.load(
            V + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )
        acc += tl.dot(p, v)

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < cur_batch_seq_len)


def token_att_fwd2(prob, value, output, b_start_loc, b_seq_len, max_input_len):
    BLOCK = 128  # Define BLOCK size based on GPU capability

    Lp, Lv = prob.shape[-1], value.shape[-1]
    assert Lp == Lv

    batch, head = b_seq_len.shape[0], prob.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lv <= 64 else 8

    _fwd_kernel_token_att2[grid](
        prob, value, output,
        b_start_loc, b_seq_len,
        prob.stride(0), prob.stride(1),
        value.stride(0), value.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK, BLOCK_DMODEL=Lv, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1
    )

# Example usage
batch_size = 2
heads = 4
seq_len = 128
d_model = 64

prob = torch.rand((batch_size, heads, seq_len, seq_len), dtype=torch.float32, device='cuda')
value = torch.rand((batch_size, heads, seq_len, d_model), dtype=torch.float32, device='cuda')
output = torch.zeros((batch_size, heads, seq_len, d_model), dtype=torch.float32, device='cuda')

b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device='cuda')

token_att_fwd2(prob, value, output, b_start_loc, b_seq_len, seq_len)
