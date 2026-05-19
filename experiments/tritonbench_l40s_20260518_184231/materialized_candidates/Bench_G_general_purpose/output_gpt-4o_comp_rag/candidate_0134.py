import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}),
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 16}),
        triton.Config({'num_warps': 32}),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr,
    initial_state_ptr, final_state_ptr,
    BT, BK, BV, NT,
    K, V,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    k_offset = i_bh * BT * BK + i_k * BK
    v_offset = i_bh * BT * BV + i_v * BV
    d_offset = i_bh * BT * BK + i_k * BK
    v_new_offset = i_bh * BT * BV + i_v * BV

    k_block_ptr = tl.make_block_ptr(k_ptr + k_offset, shape=(BT, BK), strides=(BK, 1), offsets=(0, 0))
    v_block_ptr = tl.make_block_ptr(v_ptr + v_offset, shape=(BT, BV), strides=(BV, 1), offsets=(0, 0))
    d_block_ptr = tl.make_block_ptr(d_ptr + d_offset, shape=(BT, BK), strides=(BK, 1), offsets=(0, 0))
    v_new_block_ptr = tl.make_block_ptr(v_new_ptr + v_new_offset, shape=(BT, BV), strides=(BV, 1), offsets=(0, 0))

    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + i_bh * V)
    else:
        b_h = tl.zeros((V,), dtype=tl.float32)

    for t in range(NT):
        k = tl.load(k_block_ptr + t * BK)
        v = tl.load(v_block_ptr + t * BV)
        d = tl.load(d_block_ptr + t * BK)

        h = tl.dot(k, v)
        b_h += h

        v_new = v + d * k
        tl.store(v_new_block_ptr + t * BV, v_new)

    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + i_bh * V, b_h)


def chunk_fwd_h_fn(k, v, d, initial_state=None, final_state=None):
    BT, BK = k.shape
    _, BV = v.shape
    NT = BT

    k_ptr = k.data_ptr()
    v_ptr = v.data_ptr()
    d_ptr = d.data_ptr()
    v_new = torch.empty_like(v)
    v_new_ptr = v_new.data_ptr()

    initial_state_ptr = initial_state.data_ptr() if initial_state is not None else 0
    final_state_ptr = final_state.data_ptr() if final_state is not None else 0

    USE_INITIAL_STATE = initial_state is not None
    STORE_FINAL_STATE = final_state is not None

    grid = (BK, BV, BT)

    chunk_delta_rule_fwd_kernel_h[grid](
        k_ptr, v_ptr, d_ptr, v_new_ptr,
        initial_state_ptr, final_state_ptr,
        BT, BK, BV, NT,
        BK, BV,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        STORE_FINAL_STATE=STORE_FINAL_STATE
    )

    return v_new
