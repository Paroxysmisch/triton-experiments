import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    q, k, B_Loc, B_Start_Loc, B_Seqlen, att_out,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_obs, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None]

    q_vec = tl.load(q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k_vec = tl.load(k + off_k, mask=offs_n[None, :] < cur_batch_seq_len, other=0.0)

    att_score = tl.dot(q_vec, k_vec)
    att_score *= 1.0 / (BLOCK_DMODEL**0.5)

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_n[None, :]
    )
    tl.store(att_out + off_o, att_score, mask=offs_m[:, None] < cur_batch_seq_len)


def token_att_fwd(q, k, B_Loc, B_Start_Loc, B_Seqlen, max_input_len, att_out):
    BLOCK = 128 if CUDA_CAPABILITY[0] >= 8 else 64

    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}

    batch, head = B_Seqlen.shape[0], q.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel_token_att1[grid](
        q, k, B_Loc, B_Start_Loc, B_Seqlen, att_out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        att_out.stride(0), att_out.stride(1),
        BLOCK_M=BLOCK, BLOCK_DMODEL=Lk, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1
    )
