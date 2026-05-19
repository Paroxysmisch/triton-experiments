import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att1(
    Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out,
    max_input_len, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_bl, stride_bs, stride_sl,
    stride_ab, stride_ah, stride_al,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads = tl.num_programs(axis=1)
    batch_id = pid // num_heads
    head_id = pid % num_heads

    start_loc = tl.load(B_Start_Loc + batch_id * stride_bs)
    seq_len = tl.load(B_Seqlen + batch_id * stride_sl)

    # Grid size
    n_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N

    # Offset calculations
    offset_q = batch_id * stride_qb + head_id * stride_qh
    offset_k = batch_id * stride_kb + head_id * stride_kh

    # Iterate over blocks
    for block_id in range(n_blocks):
        n_start = block_id * BLOCK_N
        n_end = min(n_start + BLOCK_N, seq_len)

        # Load Q and K blocks
        q_ptrs = Q + offset_q + tl.arange(0, BLOCK_N)[:, None] * stride_qd
        k_ptrs = K + offset_k + tl.arange(0, BLOCK_N)[None, :] * stride_kd

        q = tl.load(q_ptrs, mask=tl.arange(0, BLOCK_N) < seq_len, other=0.0)
        k = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_N) < seq_len, other=0.0)

        # Compute attention scores
        att_scores = tl.dot(q, k) * sm_scale

        # Write back results
        att_out_ptrs = Att_Out + batch_id * stride_ab + head_id * stride_ah + (start_loc + n_start) * stride_al
        tl.store(att_out_ptrs, att_scores, mask=tl.arange(0, BLOCK_N) < seq_len)

def token_att_fwd(Q, K, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    assert Q.shape == K.shape, "Query and Key must have the same shape"
    assert Q.shape[-1] == K.shape[-1], "Query and Key must have the same last dimension size"

    batch_size, num_heads, seq_len, dim = Q.shape

    BLOCK_N = 128  # Block size for sequences
    sm_scale = 1.0 / (dim ** 0.5)

    # Initialize output tensor
    Att_Out = torch.empty((batch_size, num_heads, seq_len, seq_len), device=Q.device, dtype=Q.dtype)

    # Launch kernel
    grid = (batch_size * num_heads, )
    num_warps = 4

    _fwd_kernel_token_att1[grid](
        Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out,
        max_input_len, sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        B_Loc.stride(0), B_Start_Loc.stride(0), B_Seqlen.stride(0),
        Att_Out.stride(0), Att_Out.stride(1), Att_Out.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )

    return Att_Out
