import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_cosbs, stride_cosd,
    stride_sinbs, stride_sind,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)

    offs_qh = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    offs_qs = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_k = offs_qh * stride_qh + offs_qs * stride_qs
    mask_k = (offs_qh < tl.num_programs(0)) & (offs_qs < tl.num_programs(1))

    offs_q = offs_k * stride_qd + offs_d
    offs_k = offs_k * stride_kd + offs_d

    q0 = tl.load(Q + offs_q, mask=mask_k, other=0.0)
    q1 = tl.load(Q + offs_q + BLOCK_DMODEL, mask=mask_k, other=0.0)

    cos0 = tl.load(Cos + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)
    cos1 = tl.load(Cos + stride_cosbs + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)

    sin0 = tl.load(Sin + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)
    sin1 = tl.load(Sin + stride_sinbs + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)

    out0 = q0 * cos0 - q1 * sin0
    tl.store(Q + offs_q, out0, mask=mask_k)
    out1 = q0 * sin1 + q1 * cos1
    tl.store(Q + offs_q + BLOCK_DMODEL, out1, mask=mask_k)

    k0 = tl.load(K + offs_k, mask=mask_k, other=0.0)
    k1 = tl.load(K + offs_k + BLOCK_DMODEL, mask=mask_k, other=0.0)

    cos0 = tl.load(Cos + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)
    cos1 = tl.load(Cos + stride_cosbs + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)

    sin0 = tl.load(Sin + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)
    sin1 = tl.load(Sin + stride_sinbs + offs_d, mask=offs_d < BLOCK_DMODEL, other=0.0)

    out0 = k0 * cos0 - k1 * sin0
    tl.store(K + offs_k, out0, mask=mask_k)
    out1 = k0 * sin1 + k1 * cos1
    tl.store(K + offs_k + BLOCK_DMODEL, out1, mask=mask_k)


@torch.no_grad()
def rotary_emb_fwd(Q, Cos, Sin):
    assert Q.shape[-1] == Cos.shape[-1] == Sin.shape[-1]
    assert Q.ndim == 3
    BLOCK_HEAD = 4
    BLOCK_SEQ = 32
    BLOCK_DMODEL = triton.next_power_of_2(Q.shape[-1] // 2)
    num_heads = Q.shape[1]
    num_seq = Q.shape[2]
    grid = (num_heads, num_seq)
    num_warps = 4
    if BLOCK_DMODEL >= 8192:
        num_warps = 8
    elif BLOCK_DMODEL >= 4096:
        num_warps = 4
    _rotary_kernel[grid](
        Q,
        Cos,
        Sin,
        stride_qbs=Q.stride(0) * Q.stride(1),
        stride_qh=Q.stride(1),
        stride_qd=Q.stride(2),
        stride_cosbs=Cos.stride(0),
        stride_cosd=Cos.stride(1),
        stride_sinbs=Sin.stride(0),
        stride_sind=Sin.stride(1),
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )
