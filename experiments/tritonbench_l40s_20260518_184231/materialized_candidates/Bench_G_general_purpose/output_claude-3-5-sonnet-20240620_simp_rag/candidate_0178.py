import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, K, B_Loc, B_Start_Loc, B_Seqlen, 
    att_out,
    stride_qbs, stride_qh, stride_kbs, stride_kh,
    stride_obs, stride_oh,
    max_input_len,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # Load batch-specific information
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = tl.load(B_Start_Loc + cur_batch)

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Calculate input offsets
    q_offset = (cur_batch_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    k_offset = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None]

    # Load query
    q = tl.load(Q + q_offset, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Main loop
    for start_n in range(0, max_input_len, BLOCK_N):
        # Load key
        k = tl.load(
            K + k_offset + (cur_batch_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0
        )

        # Compute attention scores
        att_scores = tl.dot(q, k)
        att_scores = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), att_scores, float("-inf"))

        # Scale attention scores
        att_scores *= (1.0 / (BLOCK_DMODEL ** 0.5))

        # Compute softmax
        att_probs = tl.softmax(att_scores, axis=1)

        # Update accumulator
        acc += tl.dot(att_probs, k.transpose())

    # Store output
    out_offset = (cur_batch_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    tl.store(att_out + out_offset, acc, mask=offs_m[:, None] < cur_batch_seq_len)

def token_att_fwd(q, k, b_loc, b_start_loc, b_seqlen, max_input_len, att_out):
    BLOCK = 128 if triton.cdiv(max_input_len, 128) <= 256 else 64

    batch, head = b_seqlen.shape[0], q.shape[1]
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk, "Query and Key dimensions must match"
    assert Lk in {16, 32, 64, 128, 256}, "Key dimension must be one of 16, 32, 64, 128, or 256"

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel_token_att1[grid](
        q, k, b_loc, b_start_loc, b_seqlen, 
        att_out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        att_out.stride(0), att_out.stride(1),
        max_input_len,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
