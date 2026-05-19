import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att1(
    Q,
    K,
    B_Loc,
    B_Start_Loc,
    B_Seqlen,
    sm_scale,
    Att_Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_att_bs,
    stride_att_h,
    stride_att_m,
    stride_att_n,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    block_index = tl.program_id(2)
    
    num_blocks_n = (MAX_SEQ_LEN + BLOCK_N - 1) // BLOCK_N
    block_m = block_index // num_blocks_n
    block_n = block_index % num_blocks_n

    batch_start = tl.load(B_Start_Loc + cur_batch)
    seq_len = tl.load(B_Seqlen + cur_batch)

    offs_m = block_m * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_q = batch_start + offs_m
    offs_k = batch_start + offs_n

    q_ptrs = Q + (offs_q[:, None] * stride_qbs + cur_head * stride_qh + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd)
    k_ptrs = K + (offs_k[None, :] * stride_kbs + cur_head * stride_kh + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd)

    mask_q = offs_m < seq_len
    mask_k = offs_n < seq_len

    q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=mask_k[None, :], other=0.0)

    att = tl.dot(q, k, allow_tf32=False)
    att *= sm_scale

    offs_att = (cur_batch * stride_att_bs + 
                cur_head * stride_att_h + 
                offs_m[:, None] * stride_att_m + 
                offs_n[None, :] * stride_att_n)
    att_ptrs = Att_Out + offs_att
    tl.store(att_ptrs, att, mask=mask_q[:, None] & mask_k[None, :])

def token_att_fwd(Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out, max_input_len):
    assert Q.is_cuda and K.is_cuda and B_Loc.is_cuda and B_Start_Loc.is_cuda and B_Seqlen.is_cuda
    assert Q.shape[0] == K.shape[0], "Number of tokens must match for Q and K"
    assert Q.shape[2] == K.shape[2], "Feature dimension must match for Q and K"
    d_k = K.shape[2]
    sm_scale = 1.0 / (d_k ** 0.5)

    if torch.cuda.get_device_capability(Q.device)[0] >= 8:
        BLOCK_N = 128
    else:
        BLOCK_N = 64

    batch = B_Seqlen.shape[0]
    num_heads = Q.shape[1]
    max_seq_len = max_input_len

    num_blocks_n = (max_seq_len + BLOCK_N - 1) // BLOCK_N
    num_blocks_m = (max_seq_len + BLOCK_N - 1) // BLOCK_N
    grid = (batch, num_heads, num_blocks_m * num_blocks_n)

    num_warps = 4 if d_k <= 64 else 8

    _fwd_kernel_token_att1[grid](
        Q, K, B_Loc, B_Start_Loc, B_Seqlen, sm_scale, Att_Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        Att_Out.stride(0), Att_Out.stride(1), Att_Out.stride(2), Att_Out.stride(3),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=d_k,
        MAX_SEQ_LEN=max_seq_len,
        num_warps=num_warps,
        num_stages=1,
    )
