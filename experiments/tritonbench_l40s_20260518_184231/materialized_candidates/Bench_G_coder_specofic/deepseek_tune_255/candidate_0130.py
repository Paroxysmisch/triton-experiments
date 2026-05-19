import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"V_BLOCK_SIZE": 256, "NUM_WARPS": 1}, num_stages=3, num_warps=1),
        triton.Config({"V_BLOCK_SIZE": 256, "NUM_WARPS": 2}, num_stages=3, num_warps=2),
        triton.Config({"V_BLOCK_SIZE": 256, "NUM_WARPS": 4}, num_stages=3, num_warps=4),
        triton.Config({"V_BLOCK_SIZE": 256, "NUM_WARPS": 8}, num_stages=3, num_warps=8),
        triton.Config({"V_BLOCK_SIZE": 256, "NUM_WARPS": 16}, num_stages=3, num_warps=16),
        triton.Config({"V_BLOCK_SIZE": 256, "NUM_WARPS": 32}, num_stages=3, num_warps=32),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    v_new,
    h,
    initial_state,
    final_state,
    s_k_h,
    s_k_t,
    s_k_d,
    s_k_bk,
    s_k_bv,
    s_v_h,
    s_v_t,
    s_v_d,
    s_v_bk,
    s_v_bv,
    s_d_h,
    s_d_t,
    s_d_d,
    s_d_bk,
    NT,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_D: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BT, BK, BV], dtype=tl.float32)
    b_h_cumsum = tl.zeros([BT, BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * BK * BV, (BK, BV), (BK, 1), (i_k * BK, i_v * BV), (BT, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (BK, s_k_t), (s_k_d, s_k_bk), (i_k * BK, i_t * s_k_d), (BK, s_k_bk), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (s_v_t, BV), (s_v_d, s_v_bk), (i_t * s_v_d, i_v * BV), (s_v_bk, BV), (0, 1))
        p_d = tl.make_block_ptr(d + i_bh * s_d_h, (BK, s_d_t), (s_d_d, s_d_bk), (i_k * BK, i_t * s_d_d), (BK, s_d_bk), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if HAS_D:
            b_d = tl.load(p_d, boundary_check=(0, 1))

        b_h *= b_d
        b_h = tl.dot(b_h, b_k, allow_tf32=False).to(b_h.dtype)
        b_h_cumsum += b_v

        p_h = tl.make_block_ptr(h + i_bh * s_k_h, (BK, s_k_t), (s_k_d, s_k_bk), (i_k * BK, i_t * s_k_d), (BK, s_k_bk), (1, 0))
        p_v_new = tl.make_block_ptr(v_new + i_bh * s_v_h, (s_v_t, BV), (s_v_d, s_v_bk), (i_t * s_v_d, i_v * BV), (s_v_bk, BV), (0, 1))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(final_state + i_bh * BK * BV, (BK, BV), (BK, 1), (i_k * BK, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_h, b_h_cumsum.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h_fn(k, v, d, initial_state=None, final_state=None, max_warps=32):
    BT, BK, BV = k.shape[0], k.shape[1], v.shape[1]
    NT, NK, NV = d.shape[1], k.shape[1], v.shape[0]
    V_BLOCK_SIZE = triton.next_power_of_2(NV)
    num_warps = min(max_warps, V_BLOCK_SIZE)
    num_warps = min(num_warps, 32)
    num_stages = 1

    v_new = torch.empty_like(v)
    h = torch.empty_like(k)
    grid = lambda META: (triton.cdiv(NK, META["BK"]), triton.cdiv(NV, META["BV"]), META["BT"])
    chunk_delta_rule_fwd_kernel_h[grid](
        k,
        v,
        d,
        v_new,
        h,
        initial_state,
        final_state,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
