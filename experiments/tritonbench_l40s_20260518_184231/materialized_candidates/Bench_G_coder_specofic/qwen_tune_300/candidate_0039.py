import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, sm_scale, req_to_tokens, b_req_idx, B_Start_Loc, B_Seqlen,
    Out,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b_req_idx_b, stride_b_req_idx_s,
    stride_b_start_loc_b, stride_b_start_loc_s,
    stride_b_seq_len_b, stride_b_seq_len_s,
    H: tl.constexpr, BLOCK_DMODEL: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)    
    
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_b_seq_len_b)
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch * stride_b_req_idx_b + (start_m + BLOCK_M - 1) * stride_b_req_idx_s)    
    cur_batch_start_index = 0
    cur_batch_start_index = tl.load(B_Start_Loc + cur_batch * stride_b_start_loc_b)
    
    block_start_loc = BLOCK_M * start_m
    
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh) * stride_qd + offs_m[:, None] * stride_qd + offs_d[None, :]
    off_k = (cur_head * stride_kh) * stride_kd + offs_n[None, :] * stride_kd + offs_d[:, None]
    off_v = (cur_head * stride_vh) * stride_vd + offs_n[:, None] * stride_vd + offs_d[None, :]
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    
    q = (q * sm_scale).to(tl.float16)
    k_ptrs = K + off_k
    v_ptrs = V + off_v    
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    
    
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_start_index + start_n) * stride_req_to_tokens, 
                    mask=(cur_batch_start_index + start_n + offs_n[None, :]) < cur_batch_seq_len, 
                    other=0.0)        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)        
        qk = tl.where(offs_m[:, None] >= (cur_batch_start_index + start_n + offs_n[None, :]), qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]        
        v = tl.load(v_ptrs + (cur_batch_start_index + start_n) * stride_req_to_tokens, 
                    mask=(cur_batch_start_index + start_n + offs_n[:, None]) < cur_batch_seq_len, 
                    other=0.0)        
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    
    off_o = (cur_batch * stride_obs + cur_head * stride_oh) * stride_od + offs_m[:, None] * stride_od + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return

class ContextAttentionFunctionPPLInt8KV(Function):
    @staticmethod
    def forward(ctx, q, k, v, o, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, max_input_len, logit_scale=1.0, train=True):
        BLOCK = 128 if not IS_TESLA else 64
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        sm_scale = (logit_scale * (2.0 / Lk) ** 0.5).to(tl.float32)
        
        batch, head = b_req_idx.shape[0], q.shape[1]
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel_int8kv[grid](
            q, k, v, sm_scale, 
            req_to_tokens, b_req_idx, b_start_loc, b_seq_len,
            o,
            req_to_tokens.stride(0), req_to_tokens.stride(1),
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            b_req_idx.stride(0), b_req_idx.stride(1),
            b_start_loc.stride(0), b_start_loc.stride(1),
            b_seq_len.stride(0), b_seq_len.stride(1),
            BLOCK_DMODEL=Lk,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

def context_attention_fwd_ppl_int8kv(q: Tensor, k: Tensor, v: Tensor, o: Tensor, 
                                   req_to_tokens: Tensor, b_req_idx: Tensor, 
                                   b_start_loc: Tensor, b_seq_len: Tensor, 
                                   max_input_len: int, logit_scale: float = 1.0, 
                                   train: bool = True):
    assert b_req_idx.is_contiguous()
    assert b_seq_len.is_contiguous()
    assert b_start_loc.is_contiguous()
    
    if (logit_scale != 1.0) and (logit_scale != 0.0):
        logit_scale = logit_scale * (2.0 ** -16.0)
    
    ContextAttentionFunctionPPLInt8KV.apply(q, k, v, o, req_to_tokens, b_req_idx, b_start_loc, b_seq_len, max_input_len, logit_scale, train)
    return
