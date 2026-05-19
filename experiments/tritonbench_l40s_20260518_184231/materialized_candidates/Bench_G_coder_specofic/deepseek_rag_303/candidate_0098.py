import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q,  # [batch, head, seq, dim]
    K,
    Cos,  # [batch, head, seq, dim//2]
    Sin,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_cosb, stride_cosh, stride_cose, stride_cose2,
    stride_sinb, stride_sinh, stride_ sine2,
    seq_lens,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    # [batch, head, seq]
    offs_head = pid_batch * stride_qb + pid_head * stride_qh
    
    offs_seq_len = pid_batch 
    seq_len = tl.load(seq_lens + offs_seq_len)
    
    offs_q0 = offs_head + tl.arange(0, BLOCK_SEQ) 
    offs_k0 = offs_head + tl.arange(0, BLOCK_SEQ) 
    
    offs_q1 = offs_head + stride_qh + tl.arange(0, BLOCK_SEQ) 
    offs_k1 = offs_head + stride_qh + tl.arange(0, BLOCK_SEQ) 
    
    offs_cos0 = offs_head + tl.arange(0, BLOCK_SEQ) * 2 
    offs_cos1 = offs_head + tl.arange(0, BLOCK_SEQ) * 2 + 1 
    
    offs_sin0 = offs_head + tl.arange(0, BLOCK_SEQ) * 2 
    offs_sin1 = offs_head + tl.arange(0, BLOCK_SEQ) * 2 + 1
    
    # [2, block_seq]
    range_2N = tl.arange(0, BLOCK_DMODEL)
    range_2N0 = range_2N
    range_2N1 = range_2N + BLOCK_DMODEL // 2
    
    # load q0, k0; [[block_seq], 2]
    q0 = tl.load(
        Q + offs_q0[:, None] * stride_qs + range_2N0[None, :] * stride_qd,
        mask=(offs_q0[:, None] < seq_len) & (range_2N0[None, :] < BLOCK_DMODEL // 2),
        other=0.0)
    
    q1 = tl.load(
        Q + offs_q1[:, None] * stride_qs + range_2N1[None, :] * stride_qd,
        mask=(offs_q1[:, None] < seq_len) & (range_2N1[None, :] < BLOCK_DMODEL // 2),
        other=0.0)
    
    k0 = tl.load(
        K + offs_k0[:, None] * stride_ks + range_2N0[None, :] * stride_kd,
        mask=(offs_k0[:, None] < seq_len) & (range_2N0[None, :] < BLOCK_DMODEL // 2),
        other=0.0)
    
    k1 = tl.load(
        K + offs_k1[:, None] * stride_ks + range_2N1[None, :] * stride_kd,
        mask=(offs_k1[:, None] < seq_len) & (range_2N1[None, :] < BLOCK_DMODEL // 2),
        other=0.0)
    
    # load [block_seq], 2]
    cos0 = tl.load(
        Cos + offs_cos0[:, None] * stride_cosb + offs_cos0[:, None] * stride_cose2,
        mask=offs_cos0 < seq_len,
        other=0.0)
    
    cos1 = tl.load(
        Cos + offs_cos1[:, None] * stride_cosb + offs_cos1[:, None] * stride_cose2,
        mask=offs_cos1 < seq_len,
        other=0.0)
    
    sin0 = tl.load(
        Sin + offs_sin0[:, None] * stride_sinb + offs_sin0[:, None] * stride_sine2,
        mask=offs_sin0 < seq_len,
        other=0.0)
    
    sin1 = tl.load(
        Sin + offs_sin1[:, None] * stride_sinb + offs_sin1[:, None] * stride_sine2,
        mask=offs_sin1 < seq_len,
        other=0.0)
    
    # [2, block_seq]
    
    out0 = q0 * cos0 - q1 * sin0
    tl.store(
        Q + offs_q0[:, None] * stride_qs + range_2N0[None, :] * stride_qd,
        out0,
        mask=(offs_q0[:, None] < seq_len) & (range_2N0[None, :] < BLOCK_DMODEL // 2))
    
    out1 = q0 * sin1 + q1 * cos1
    tl.store(
        Q + offs_q1[:, None] * stride_qs + range_2N1[None, :] * stride_qd,
        out1,
        mask=(offs_q1[:, None] < seq_len) & (range_2N1[None, :] < BLOCK_DMODEL // 2))

@torch.no_grad()
def rotary_emb_fwd(Q, Cos, Sin, seq_len):
    """Q: [batch, head, seq, dim]"""
    
    assert Q.is_contiguous()
    assert Cos.is_contiguous()
    assert Sin.is_contiguous()
    assert Q.dim() == 4
    assert Q.size(0) == Cos.size(0)
    assert Q.size(2) == seq_len.item()
    assert Cos.size(2) == seq_len.item() * 2
    assert Q.size(-1) % 2 == 0  # for even dim
    dim = Q.size(-1) // 2  # dim(model) should be 2n
    
    BLOCK_HEAD = 1
    grid = (Q.size(0), Q.size(1), dim)
    num_warps = 4
    if grid[-1] <= 128:
        num_warps = 8
    if grid[-1] <= 64:
        num_warps = 16
        
    _rotary_kernel[grid](
        Q, Cos, Sin, stride_qb=Q.stride(0), stride_qh=Q.stride(1), stride_qs=Q.stride(2), stride_qd=Q.stride(3),
        stride_kb=Q.stride(0), stride_kh=Q.stride(1), stride_ks=Q.stride(2), stride_kd=Q.stride(3),
        stride_cosb=Cos.stride(0), stride_cosh=Cos.stride(1), stride_cose=Cos.stride(2), stride_cose2=Cos.stride(3),
        stride_sin
