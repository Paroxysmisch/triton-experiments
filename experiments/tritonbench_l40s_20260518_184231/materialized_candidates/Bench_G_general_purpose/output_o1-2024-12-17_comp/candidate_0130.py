import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"]
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, h_ptr, v_new_ptr,
    initial_state_ptr, final_state_ptr,
    stride_kBT, stride_kBK, stride_vBT, stride_vBK,
    stride_dBT, stride_dBK, stride_hBT, stride_hBK,
    stride_vnewBT, stride_vnewBK,
    stride_initBT, stride_finalBT,
    BT, BK, BV, NT,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    off_k = i_k * BLOCK_K
    off_v = i_v * BLOCK_V
    off_bh = i_bh

    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + off_bh * stride_initBT, mask=(off_bh < BT))
    else:
        b_h = tl.zeros([BLOCK_V], dtype=tl.float32)

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + off_bh * stride_kBT + off_k * stride_kBK,
        shape=(BK, 1),
        strides=(stride_kBK, 1),
        sizes=(BK, 1),
        block_shape=(BLOCK_K, 1),
        order=(0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + off_bh * stride_vBT + off_v * stride_vBK,
        shape=(BV, 1),
        strides=(stride_vBK, 1),
        sizes=(BV, 1),
        block_shape=(BLOCK_V, 1),
        order=(0, 1)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + off_bh * stride_dBT + off_k * stride_dBK,
        shape=(BK, 1),
        strides=(stride_dBK, 1),
        sizes=(BK, 1),
        block_shape=(BLOCK_K, 1),
        order=(0, 1)
    )
    vnew_block_ptr = tl.make_block_ptr(
        base=v_new_ptr + off_bh * stride_vnewBT + off_v * stride_vnewBK,
        shape=(BV, 1),
        strides=(stride_vnewBK, 1),
        sizes=(BV, 1),
        block_shape=(BLOCK_V, 1),
        order=(0, 1)
    )

    for _ in range(NT):
        k_val = tl.load(k_block_ptr, mask=(off_k + tl.arange(0, BLOCK_K) < BK))
        d_val = tl.load(d_block_ptr, mask=(off_k + tl.arange(0, BLOCK_K) < BK))
        v_val = tl.load(v_block_ptr, mask=(off_v + tl.arange(0, BLOCK_V) < BV))

        tl.store(h_ptr + (off_bh * stride_hBT + off_v * stride_hBK), b_h, mask=(off_v + tl.arange(0, BLOCK_V) < BV))

        update = tl.sum(k_val * d_val, axis=0)
        b_h += update

        new_v_val = v_val + b_h
        tl.store(v_block_ptr, new_v_val, mask=(off_v + tl.arange(0, BLOCK_V) < BV))
        tl.store(vnew_block_ptr, new_v_val, mask=(off_v + tl.arange(0, BLOCK_V) < BV))

        k_block_ptr = tl.advance(k_block_ptr, 1, 0)
        d_block_ptr = tl.advance(d_block_ptr, 1, 0)
        v_block_ptr = tl.advance(v_block_ptr, 1, 0)
        vnew_block_ptr = tl.advance(vnew_block_ptr, 1, 0)

    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + off_bh * stride_finalBT, b_h, mask=(off_bh < BT))


def chunk_fwd_h_fn(
    k, v, d, h, v_new,
    initial_state=None, final_state=None,
    NT=1, USE_INITIAL_STATE=False, STORE_FINAL_STATE=False
):
    import math
    BT, BK = k.shape[0], k.shape[1]
    BV = v.shape[1]

    stride_kBT, stride_kBK = k.stride(0), k.stride(1)
    stride_vBT, stride_vBK = v.stride(0), v.stride(1)
    stride_dBT, stride_dBK = d.stride(0), d.stride(1)
    stride_hBT, stride_hBK = h.stride(0), h.stride(1)
    stride_vnewBT, stride_vnewBK = v_new.stride(0), v_new.stride(1)

    stride_initBT = initial_state.stride(0) if initial_state is not None else 0
    stride_finalBT = final_state.stride(0) if final_state is not None else 0

    BLOCK_K = 32
    BLOCK_V = 32

    grid_k = math.ceil(BK / BLOCK_K)
    grid_v = math.ceil(BV / BLOCK_V)
    grid = (grid_k, grid_v, BT)

    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, h, v_new,
        initial_state if initial_state is not None else 0
