import torch
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_b0z, stride_b0h, stride_b0m, stride_b0n,
    Z, H,  N_CTX: tl.constexpr, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_z * stride_qz + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    off_k = off_z * stride_kz + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    off_v = off_z * stride_vz + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    b0_ptrs = B0 + off_hz * stride_b0h + offs_n[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        b0 = tl.load(b0_ptrs + start_n * stride_b0n)
        qk = qk + b0
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(m_ij)
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale
        acc_scale = l_i / l_i_new * alpha
        tl.store(b0_ptrs + start_n * stride_b0n, m_ij + m_i_new)
        acc_scale_ptrs = b0_ptrs + stride_b0h * H + stride_b0n * N_CTX + off_z * stride_b0z + off_h * stride_b0h + (offs_m[:, None] * stride_b0m + offs_n[None, :])
        tl.store(acc_scale_ptrs, acc_scale)
        v = tl.load(v_ptrs + start_n * stride_vn)
        p = p.to(tl.float16)
        acc_scale = acc_scale.to(tl.float16)
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_o = off_z * stride_oz + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

@torch.inference_mode()
def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, block_size, bias_last_dim, q_last_dim, out_dtype):
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] in {16, 32, 64, 128, 256, 512}
    assert q.shape[-2] == k.shape[-2]
    assert q.shape[-2] % triton.next_power_of_2(block_size) == 0
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2] == rel_h_w.shape[:-2]
    assert q.dtype == k.dtype == v.dtype == rel_h_w.dtype
    assert q.dtype in {torch.float16, torch.bfloat16}
    assert q_last_dim in {"heads", "tokens"}
    if bias_last_dim:
        assert rel_h_w.shape[-1] == 1
    else:
        assert rel_h_w.shape[-1] == q.shape[-2]
    B, H, N = q.shape[:-2], q.shape[-2], q.shape[-1]
    OUT_DTYPE = q.dtype if out_dtype is None else out_dtype
    N_CTX = N // 2
    grid = (triton.cdiv(N_CTX, block_size), B[-1] * H)
    rel_h_w = rearrange(rel_h_w, "z h ... -> z h (...)").contiguous()
    bias_last_size = rel_h_w.numel() // len(B) // H
    static_shapes = {"heads": (B, H, N, N), "tokens": (B, N, N, bias_last_size)}
    static_stride = {"heads": (rel_h_w.stride(1), rel_h_w.stride(2), rel_h_w.stride(3), 1), "tokens": (rel_h_w.stride(2), rel_h_w.stride(3), 1, bias_last_size)}
    rel_h_w_strides = static_stride[q_last_dim]
    with torch.cuda.device(q.device.index):
        _fwd_kernel_aligned[grid](
            q, k, v, rel_h_w, sm_scale, torch.empty_like(q, dtype=OUT_DTYPE),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            rel_h_w.stride(0), rel_h_w.stride(1), rel_h_w.stride(2), rel_h_w.stride(3),
            rel_h_w_strides[0], rel_h_w_strides[1], rel_h_w_strides[2], rel_h_w_strides[3],
            B[0], B[1], N_CTX,
            N, q.shape[-1],
            BLOCK_M=block_size,
            BLOCK_N=block_size,
            BLOCK_DMODEL=q.shape[-1],
            num_warps=4,
            num_stages=2,
        )
    return

def attention_rel_h_rel_w_kernel_aligned(q, k, v, rel_h, rel_w, sm_scale, block_size, bias_last_dim, q_last_dim, out_dtype):
    rel_hw = torch.cat((rel_h, rel_w), dim=-1).to(q.dtype)
    _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_hw, sm_scale, block_size, bias_last_dim, q_last_dim, out_dtype)
    return
