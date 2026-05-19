import torch
import triton
import triton.language as tl
from packaging import version

@triton.autotune(
    configs=[
        triton.Config({"BT": 1}, num_warps=1),
        triton.Config({"BT": 1}, num_warps=2),
        triton.Config({"BT": 1}, num_warps=4),
        triton.Config({"BT": 1}, num_warps=8),
        triton.Config({"BT": 1}, num_warps=16),
        triton.Config({"BT": 1}, num_warps=32),
    ],
    key=["BK", "BV"],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k, v, d, initial_state, final_state, h, v_new, initial_state_b, initial_state_k, store_final_state,
    BT, BK, BV, H, K, V, NT, NH, NDK: tl.constexpr, NDV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, CHECK: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    BV = min(BV, V // NH)
    BK = min(BK, K)
    i_h = i_bh * NH + i_k

    # block ptr
    bp_k = tl.make_block_ptr(
        base=k + i_bh * K,
        shape=(K, NT),
        strides=(1, NDK),
        offsets=(0, 0),
        block_shape=(BK, BT),
        order=(1, 0),
    )
    bp_v = tl.make_block_ptr(
        base=v + i_bh * V,
        shape=(V, NT),
        strides=(1, NDV),
        offsets=(0, 0),
        block_shape=(BV, BT),
        order=(1, 0),
    )
    bp_d = tl.make_block_ptr(
        base=d + i_bh * K,
        shape=(K, NT),
        strides=(NDK, 1),
        offsets=(0, 0),
        block_shape=(BK, BT),
        order=(0, 1),
    )
    bp_h = tl.make_block_ptr(
        base=h + (i_bh * NH + i_k) * NT,
        shape=(NT, V),
        strides=(V, 1),
        offsets=(0, 0),
        block_shape=(BT, BV),
        order=(1, 0),
    )
    bp_v_new = tl.make_block_ptr(
        base=v_new + i_bh * V,
        shape=(V, NT),
        strides=(1, NDV),
        offsets=(0, 0),
        block_shape=(BV, BT),
        order=(1, 0),
    )

    # get acc
    if USE_INITIAL_STATE:
        b_h = tl.load(
            initial_state + i_k * initial_state_k + i_bh * initial_state_b,
            mask=(i_k < H) & (i_bh < NH),
            other=0,
        ).to(tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)

    for i in range(0, NT, BT):
        b_k = tl.load(bp_k, boundary_check=(0, 1))
        b_d = tl.load(bp_d, boundary_check=(0, 1))
        b_v = tl.load(bp_v, boundary_check=(0, 1))
        # [BK, BT] * [BT, BV] = [BK, BV]
        b_h = b_h * tl.math.exp2(b_d)[:, None] + tl.dot(b_k, b_v, allow_tf32=False)

        tl.store(bp_h, b_h.to(bp_h.dtype.element_ty), boundary_check=(0, 1))
        b_v_new = b_h

        if CHECK and i == 0:
            b_v_new_check = tl.load(bp_v_new, boundary_check=(0, 1))
            assert (
                (b_v_new.to(b_v_new_check.dtype) - b_v_new_check) == 0
            ).all(), "initial state is wrong"

        bp_k = tl.advance(bp_k, (0, BT))
        bp_v = tl.advance(bp_v, (0, BT))
        bp_d = tl.advance(bp_d, (0, BT))
        bp_h = tl.advance(bp_h, (BT, 0))
        bp_v_new = tl.advance(bp_v_new, (0, BT))

    if STORE_FINAL_STATE:
        bp_final_state = tl.make_block_ptr(
            base=final_state + i_k * final_state_k + i_bh * final_state_b,
            shape=(K, V),
            strides=(V, 1),
            offsets=(0, 0),
            block_shape=(BK, BV),
            order=(1, 0),
        )
        tl.store(bp_final_state, b_h.to(bp_final_state.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h_fn(k, v, d, initial_state, final_state, BT, BK, BV, initial_state_b, initial_state_k, store_final_state):
    B, H, K, V = *k.shape, v.shape[-1]
    NT = k.shape[-2]
    NH = H

    BK = min(BK, k.shape[-1])
    BV = min(BV, v.shape[-1])

    h = k.new_empty(B, NH, NT, V)
    v_new = k.new_empty(B, H, NT, V)

    grid = (triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(NH, H))

    def run_kernel(H, NT, initial_state_b, initial_state_k):
        chunk_delta_rule_fwd_kernel_h[grid](
            k, v, d, initial_state, final_state, h, v_new, initial_state_b, initial_state_k, store_final_state,
            BT, BK, BV, H, K, V, NT, NH,
            NDK=k.stride(-2),
            NDV=v.stride(-2),
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=store_final_state,
            CHECK=initial_state is not None,
        )

    if initial_state is not None:
        run_kernel(H, NT, initial_state_b, initial_state_k)
    else:
        run_kernel(H, NT, 0, 0)

    return h, v_new
