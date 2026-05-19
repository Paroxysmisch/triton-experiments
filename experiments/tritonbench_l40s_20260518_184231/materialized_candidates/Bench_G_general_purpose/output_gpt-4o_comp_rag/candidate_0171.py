import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q,
    K,
    B_Loc,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_obs,
    stride_oh,
    sm_scale,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_m = start_m * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, Q.shape[2])

    q_ptrs = Q + cur_batch * stride_qbs + cur_head * stride_qh + offs_m[:, None] * Q.shape[2] + offs_d[None, :]
    k_ptrs = K + cur_batch * stride_kbs + cur_head * stride_kh + offs_n[None, :] * K.shape[2] + offs_d[:, None]

    q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < cur_batch_seq_len, other=0.0)

    att = tl.dot(q, k)
    att = att * sm_scale

    mask = tl.where(offs_m[:, None] < cur_batch_seq_len, 1.0, 0.0)
    att = att * mask

    att_out_ptrs = Att_Out + cur_batch * stride_obs + cur_head * stride_oh + offs_m[:, None] * Att_Out.shape[2] + offs_n[None, :]
    tl.store(att_out_ptrs, att, mask=offs_m[:, None] < cur_batch_seq_len)

def token_att_fwd(Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out, max_input_len):
    BLOCK_N = 128 if triton.cuda.current_device().compute_capability >= 80 else 64

    sm_scale = 1.0 / (Q.shape[-1] ** 0.5)
    batch, head = Q.shape[0], Q.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_N))
    num_warps = 4 if Q.shape[-1] <= 64 else 8

    _fwd_kernel_token_att1[grid](
        Q,
        K,
        B_Loc,
        B_Start_Loc,
        B_Seqlen,
        Att_Out,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        Att_Out.stride(0),
        Att_Out.stride(1),
        sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
