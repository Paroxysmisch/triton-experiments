import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out,
    Q_batch_stride, Q_head_stride, Q_dim_stride,
    K_batch_stride, K_head_stride, K_dim_stride,
    B_Loc_batch_stride, B_Loc_seq_stride,
    Att_Out_batch_stride, Att_Out_head_stride, Att_Out_seq_stride,
    max_input_len,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bat = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_blk = tl.program_id(2)
    
    start_loc = tl.load(B_Start_Loc + pid_bat)
    seq_len = tl.load(B_Seqlen + pid_bat)
    
    blk_start = pid_blk * BLOCK_N
    offs_n = blk_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seq_len
    
    # Load B_Loc to get key indices
    b_loc_offsets = pid_bat * B_Loc_batch_stride + offs_n * B_Loc_seq_stride
    key_indices = tl.load(B_Loc + b_loc_offsets, mask=mask_n, other=0)
    
    # Load query
    q_offset = pid_bat * Q_batch_stride + pid_head * Q_head_stride
    d_offsets = tl.arange(0, BLOCK_D)
    q = tl.load(Q + q_offset + d_offsets * Q_dim_stride, mask=d_offsets < Q_dim_stride, other=0.0)
    
    # Load keys
    k_offsets = key_indices[:, None] * K_batch_stride + pid_head * K_head_stride + d_offsets[None, :] * K_dim_stride
    k = tl.load(K + k_offsets, mask=mask_n[:, None] & (d_offsets[None, :] < K_dim_stride), other=0.0)
    
    # Compute attention
    att_value = tl.sum(q[None, :] * k, axis=1) * sm_scale
    
    # Store output
    out_offsets = pid_bat * Att_Out_batch_stride + pid_head * Att_Out_head_stride + offs_n * Att_Out_seq_stride
    tl.store(Att_Out + out_offsets, att_value, mask=mask_n)

def token_att_fwd(q, k, B_Loc, B_Start_Loc, B_Seqlen, Att_Out, sm_scale=None):
    assert q.is_cuda and k.is_cuda and B_Loc.is_cuda and B_Start_Loc.is_cuda and B_Seqlen.is_cuda
    assert q.shape[0] == B_Start_Loc.shape[0], "Batch size mismatch between Q and B_Start_Loc"
    assert q.shape[1] == k.shape[1], "Number of heads mismatch between Q and K"
    assert q.shape[2] == k.shape[2], "Feature dimension mismatch between Q and K"
    
    dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (dim ** 0.5)
    
    max_input_len = torch.max(B_Seqlen).item()
    
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(dim)
    if BLOCK_D < 16:
        BLOCK_D = 16
    
    num_warps = 4
    if BLOCK_D >= 2048:
        num_warps = 8
    elif BLOCK_D >= 1024:
        num_warps = 4
    
    batch, heads = q.shape[0], q.shape[1]
    grid = (batch, heads, triton.cdiv(max_input_len, BLOCK_N))
    
    _fwd_kernel_token_att1[grid](
        q, k, B_Loc, B_Start_Loc, B_Seqlen, Att_Out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        B_Loc.stride(0), B_Loc.stride(1),
        Att_Out.stride(0), Att_Out.stride(1), Att_Out.stride(2),
        max_input_len,
        sm_scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps
    )
