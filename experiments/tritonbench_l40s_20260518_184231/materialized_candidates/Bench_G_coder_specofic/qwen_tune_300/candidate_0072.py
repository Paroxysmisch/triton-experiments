import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q, k, g, A,
    s_qk_h, s_qk_t, s_qk_d,
    s_g_h, s_g_t,
    s_A_h, s_A_t, s_A_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr,
    DK: tl.constexpr,
    USE_G: tl.constexpr,
    ADD_INTER: tl.constexpr,
    CHECK: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i in range(tl.cdiv(DK, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i * BK, 0), (BK, BT), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, ), (s_g_t, ), (0, ), (BT, ), (1, ))
        p_A = tl.make_block_ptr(A + (i_bh // B) * s_A_h + i * BT * BT * H // B, (BT, BT), (s_A_t, s_A_d), (0, i_k * BT), (BT, BT), (1, 0))

        if i_k == 0:
            b_q = tl.load(q + i_bh * s_qk_h + i * BK * s_qk_d, mask=(i * BK + tl.arange(0, BK)) < DK, other=0)
            b_q = b_q.to(tl.float32)
            b_q = b_q / scale
            b_k = tl.load(k + i_bh * s_qk_h + i * BK * s_qk_d, mask=(i * BK + tl.arange(0, BK)) < DK, other=0)
            b_k = b_k.to(tl.float32)
            b_g = tl.load(g + i_bh * s_g_h, mask=True, other=0)
            b_g = b_g.to(tl.float32)
        else:
            b_q = tl.load(q + i_bh * s_qk_h + i * BK * s_qk_d, mask=(i * BK + tl.arange(0, BK)) < DK, other=0)
            b_q = b_q.to(tl.float32)
            b_k = tl.load(k + i_bh * s_qk_h + i * BK * s_qk_d, mask=(i * BK + tl.arange(0, BK)) < DK, other=0)
            b_k = b_k.to(tl.float32)
            b_g = tl.load(g + i_bh * s_g_h, mask=True, other=0)
            b_g = b_g.to(tl.float32)

        if i_k > 0 or ADD_INTER:
            b_A += tl.dot(b_q, b_k, allow_tf32=False)
        if USE_G:
            b_A = b_A * (tl.exp(b_g) * tl.exp(b_g).T)
        else:
            b_A = tl.exp(b_A + b_g.to(b_A.dtype))

        tl.store(p_A, b_A.to(A.dtype.element_ty), mask=True)
        if i_k == 0:
            b_A = tl.zeros([BT, BT], dtype=tl.float32)
        else:
            b_A = tl.log(b_A)

    if CHECK and i_k == 0:
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i in range(tl.cdiv(DK, BK)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i * BK, 0), (BK, BT), (0, 1))
            p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, ), (s_g_t, ), (0, ), (BT, ), (1, ))
            p_A = tl.make_block_ptr(A + (i_bh // B) * s_A_h + i * BT * BT * H // B, (BT, BT), (s_A_t, s_A_d), (0, i * BT), (BT, BT), (1, 0))

            b_k = tl.load(k + i_bh * s_qk_h + i * BK * s_qk_d, mask=(i * BK + tl.arange(0, BK)) < DK, other=0)
            b_k = b_k.to(tl.float32)
            b_g = tl.load(g + i_bh * s_g_h, mask=True, other=0)
            b_g = b_g.to(tl.float32)

            b_A += tl.dot(b_q, b_k, allow_tf32=False)
            if USE_G:
                b_A = b_A * (tl.exp(b_g) * tl.exp(b_g).T)
            else:
                b_A = tl.exp(b_A + b_g.to(b_A.dtype))
            tl.store(p_A, b_A.to(A.dtype.element_ty), mask=True)
            if i > 0:
                b_A = tl.log(b_A)

def chunk_fwd_intra_gated_gk_fn(q, k, g, B, H, BT, DK, B_R, scale, use_g, add_inter, device, dtype, check=False):
    T = q.shape[1]
    NT = triton.cdiv(T, BT)

    A = torch.empty((B, H, NT * BT, NT * BT), device=device, dtype=dtype)
    BK = min(64, DK)

    NT_K = 1
    if DK > BK:
        NT_K = triton.cdiv(DK, BK)

    def grid(meta): return (NT_K, NT_K, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A,
        q.stride(1), q.stride(2),
        g.stride(1),
        A.stride(2), A.stride(3),
        B, H, T, scale,
        BT=BT, BK=BK, DK=DK, USE_G=use_g, ADD_INTER=add_inter, CHECK=check,
    )
    return A
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q, k, g, A,
    s_qk_h, s_qk_t, s_qk_d,
    s_g_h, s_g_t,
    s_A_h, s_A_t, s_A_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr,
    DK: tl.constexpr,
    USE_G: tl.constexpr,
    ADD_INTER: tl.constexpr,
    CHECK: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p, i_o = i_k // BT, i_k % BT
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i in range(tl.cdiv(DK, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_p * BT, i * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i * BK, i_o * BT), (BK, BT), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, ), (s_g_t, ), (i_p * BT, ), (BT, ), (1, ))
        p_A = tl.make_block_ptr(A + (i_bh // B) * s_A_h + i * BT * BT * H // B, (BT, BT), (s_A_t, s_A_d), (i_o * BT, i_k * BT), (BT, BT), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        if USE_G:
            b_A += tl.dot(b_q, b_k, allow_tf32=False) * tl.exp(b_g)[:, None]
        else:
            b_A += tl.dot(b_q, b_k, allow_tf32=False)
            b_A = b_A * tl.exp(b_g.to(b_A.dtype))

        if ADD_INTER:
            b_A += tl.dot
