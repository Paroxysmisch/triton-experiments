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
    stride_attobs,
    stride_attoh,
    max_input_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = (
        (cur_batch_in_all_start_index + offs_n[None, :]) * stride_kbs
        + cur_head * stride_kh
        + offs_d[:, None]
    )
    off_att_out = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_attobs
        + cur_head * stride_attoh
        + offs_n[None, :]
    )

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k = tl.load(K + off_k, mask=offs_n[None, :] < cur_batch_seq_len, other=0.0)

    # compute attention scores
    qk = tl.dot(q, k)
    qk *= 1.0 / (BLOCK_DMODEL ** 0.5)

    # store attention scores
    tl.store(Att_Out + off_att_out, qk, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_n[None, :] < cur_batch_seq_len))

def token_att_fwd(q, k, b_loc, b_start_loc, b_seq_len, att_out, max_input_len):
    if CUDA_CAPABILITY[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lq in {16, 32, 64, 128, 256}

    batch, head = b_seq_len.shape[0], q.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lq <= 64 else 8

    global cached_kernel
    if cached_kernel:
        cached_kernel(
            grid,
            num_warps,
            q,
            k,
            b_loc,
            b_start_loc,
            b_seq_len,
            att_out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            att_out.stride(0),
            att_out.stride(1),
            max_input_len,
        )
        return

    _fwd_kernel_token_att1[grid](
        q,
        k,
        b_loc,
        b_start_loc,
        b_seq_len,
        att_out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        max_input_len,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lq,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    cached_kernel = wrap_kernel_launcher(_fwd_kernel_token_att1)
