import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"], 
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q, k, v, h, g, o, 
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_h_h, s_h_t, s_h_d,
    scale,
    BT: tl.constexpr, 
    BK: tl.constexpr, 
    BV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (BT, BT), (s_k_t, s_k_d), (0, i_k * BK), (BK, BT), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (BT, BT), (s_k_d, s_k_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (BT, BT), (s_h_t, s_h_d), (0, i_k * BK), (BK, BT), (1, 0))
    
    b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BK, BT], dtype=tl.float32)
    b_s = tl.zeros([BK, BT], dtype=tl.float32)
    
    for i in range(0, tl.cdiv(BT, BK)):
        p_bh = tl.make_block_ptr(h + i_bh * s_h_h, (BT, BT), (s_h_t, s_h_d), (i * BK, i_v * BV), (BK, BV), (1, 0))
        b_h_ = tl.load(p_bh, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (BT, BT), (s_v_t, s_v_d), (0, i * BK), (BT, BK), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
        b_o_ = tl.dot(b_q, b_v, allow_tf32=False)
        b_o += b_o_ * tl.exp(b_s)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (BT, BT), (s_k_t, s_k_d), ((i + 1) * BK, 0), (BK, BT), (1, 0))
    
    p_o = tl.make_block_ptr(o + (i_k * BK + i_v * BV + i_bh * s_v_h), (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_fwd_o_fn(q, k, v, h, g, BT):
    B, H, T, K = *k.shape, q.shape[-1]
    scale = K ** -0.5
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(K), 32)
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NH = triton.cdiv(H, 32)
    grid = (NK, NH, B)
    o = torch.empty(NK, B, H, K, K, device=q.device, dtype=torch.float32)
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale,
        BT=BT, 
        BK=BK, 
        BV=BV,
        num_warps=4,
    )
    return o
