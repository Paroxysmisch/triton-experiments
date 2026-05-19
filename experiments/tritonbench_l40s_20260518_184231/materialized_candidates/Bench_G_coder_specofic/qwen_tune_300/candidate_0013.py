import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,  # query, key, value, and scale
    L,  # prompt mask
    Out,  # output
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lm,  # stride for prompt mask
    Z, H,  # batch and head dimension
    N_CTX,  # context length
    kv_group_num,  # KVG head group number
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    # Kernel logic
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Step 1: compute qk^T
    # load q: it will stay in SRAM throughout
    q = tl.load(Q + cur_batch * stride_qz + cur_head * stride_qh + start_m * stride_qm + tl.arange(0, BLOCK_M),
                mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX, other=0.0)

    # compute qk^T
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # load the first block of K, V for the sake of initialization
    k = tl.load(K + cur_kv_head * stride_kh + tl.arange(0, BLOCK_N)[None, :] * stride_kn,
                mask=(tl.arange(0, BLOCK_N)[None, :] < N_CTX), other=0.0)
    v = tl.load(V + cur_kv_head * stride_vh + tl.arange(0, BLOCK_N)[:, None] * stride_vk,
                mask=(tl.arange(0, BLOCK_N)[:, None] < N_CTX), other=0.0)

    # # causal mask
    # start_n = tl.program_id(2)
    # if IS_TRITON_22:
    #     causal_mask = start_n * BLOCK_N + tl.arange(0, BLOCK_N) <= start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # else:
    #     causal_mask = (start_n * BLOCK_N + tl.arange(0, BLOCK_N)) <= (start_m * BLOCK_M + tl.arange(0, BLOCK_M))

    # # prompt mask
    # p_mask = tl.load(L + cur_batch * stride_lz + cur_head * stride_lh + start_m * stride_lm + tl.arange(0, BLOCK_M),
    #                  mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX, other=0.0)
    # p_mask = p_mask[:, None].to(tl.float32)

    # # Step 2: loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # # [update] causal mask
        # if IS_TRITON_22:
        #     causal_mask = start_n * BLOCK_N + tl.arange(0, BLOCK_N) <= start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # else:
        #     causal_mask = (start_n * BLOCK_N + tl.arange(0, BLOCK_N)) <= (start_m * BLOCK_M + tl.arange(0, BLOCK_M))

        # [update] load k, v
        k = tl.load(K + cur_kv_head * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_N)[None, :] * stride_kn,
                    mask=(start_n + tl.arange(0, BLOCK_N)[None, :]) < N_CTX, other=0.0)
        v = tl.load(V + cur_kv_head * stride_vh + start_n * stride_vk + tl.arange(0, BLOCK_N)[:, None] * stride_vk,
                    mask=(start_n + tl.arange(0, BLOCK_N)[:, None]) < N_CTX, other=0.0)

        # [update] compute qk^T
        # [update] causal attention
        # k = tl.where(causal_mask, k, float("-inf"))
        # [update] prompt attention
        # q = q * p_mask + (1 - p_mask) * float("-inf")
        qk = tl.dot(q, k)
        qk = tl.where((start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] >= (start_n + tl.arange(0, BLOCK_N))[None, :],
                      qk * sm_scale, float("-inf"))
        # [update] compute m_i, l_i
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # [update] update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # [update] update output accumulator
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # Step 3: write back l and m
    # get write mask
    write_mask = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX
    # # prompt mask
    # p_mask = tl.load(L + cur_batch * stride_lz + cur_head * stride_lh + start_m * stride_lm + tl.arange(0, BLOCK_M),
    #                  mask=write_mask, other=0.0)
    # p_mask = p_mask.to(tl.float32)
    # # causal mask
    # c_mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M) <= N_CTX
    # # Step 4: write back acc
    # # [update] causal attention
    # # acc = acc * p_mask + (1 - p_mask) * float(0)
    # # [update] prompt attention
    # # acc = acc * c_mask + float(0)
    tl.store(Out + cur_batch * stride_oz + cur_head * stride_oh + start_m * stride_om + tl.arange(0, BLOCK_M),
             acc, mask=write_mask)

class ContextAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, p, l, mask=None):
        # prepare output
        out = torch.empty_like(q)
        # reshape input data into 2D tensor
        q_2d = q.view(q.shape[0] * q.shape[1], q.shape[2])
        k_2d = k.view(k.shape[0] * k.shape[1], k.shape[2])
        v_2d = v.view(v.shape[0] * v.shape[1], v.shape[2])
        out_2d = out.view(out.shape[0] * out.shape[1], out.shape[2])
        p_2d = p.view(p.shape[0] * p.shape[1], p.shape[2])
        l_2d = l.view(l.shape[0] * l.shape[1], l.shape[2])
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // q.element_size()
        BLOCK_M = min(MAX_FUSED_SIZE, triton.next_power_of_2(q.shape[2]))
        # We only allow one block M to avoid MMA OOM
        BLOCK_M = min(BLOCK_M, 128)
        # heuristics for number of warps
        num_warps = min(max(BLOCK_M // 256, 1), 8)
        # enqueue kernel
        _fwd_kernel[(q.shape[0], q.shape[1], lambda meta: triton.cdiv(q.shape[2], meta['BLOCK_M']))](
            q_2d, k_2d, v_2d, q.scale,  # data ptrs and scaling factor
            p_2d,  # prompt mask
            out_2d,  # output
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # strides
            k.stride(0), k.stride(
