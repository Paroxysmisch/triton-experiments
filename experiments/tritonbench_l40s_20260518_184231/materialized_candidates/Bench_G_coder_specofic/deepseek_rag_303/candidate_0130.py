import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Dict, Any, Tuple

@triton.autotune(
    configs=[
        triton.Config({"BT": 16, "BK": 128, "BV": 128, "NB": 4}, num_warps=1),
        triton.Config({"BT": 32, "BK": 64, "BV": 64, "NB": 4}, num_warps=2),
        triton.Config({"BT": 32, "BK": 256, "BV": 64, "NB": 4}, num_warps=4),
        triton.Config({"BT": 32, "BK": 64, "BV": 128, "NB": 4}, num_warps=8),
        triton.Config({"BT": 64, "BK": 128, "BV": 128, "NB": 4}, num_warps=16},
        key=["BT", "BK", "BV", "NB"],
    )
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    # Tensor for "begin"
    b,
    # Input and output state tensors, dimensions [N], mostly unchanged
    initial_state,
    final_state,
    k,
    v,
    d,
    h,
    v_new,
    stride_b,
    stride_initial_state_b,
    stride_final_state_b,
    stride_k,
    stride_v,
    stride_d,
    stride_h,
    stride_v_new,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    I_bh = tl.program_id(0)
    i_k = tl.program_id(1)
    i_v = tl.program_id(2)
    p_initial_state = tl.make_block_ptr(
        base=initial_state,
        shape=BT,
        strides=stride_initial_state_b,
        offsets=I_bh * BT,
        block_shape=BT,
        order=(0,),
    )
    p_final_state = tl.make_block_ptr(
        base=final_state,
        shape=BT,
        strides=stride_final_state_b,
        offsets=I_bh * BT,
        block_shape=BT,
        order=(0,),
    )
    i_k = i_k * BK
    I_k = i_k + tl.arange(0, BK)
    i_v = i_v * BV
    I_v = i_v + tl.arange(0, BV)
    p_d = tl.make_block_ptr(
        base=d,
        shape=(BT, K),
        strides=(stride_b, stride_k),
        offsets=(I_bh * BT, i_k),
        block_shape=(BT, BK),
        order=(1, 0),
    )
    p_k = tl.make_block_ptr(
        base=k,
        shape=(BT, K, V),
        strides=(stride_b, stride_k, stride_v),
        offsets=(I_bh * BT, i_k, i_v[None, :]),
        block_shape=(BT, BK, BV),
        order=(1, 0, 2),
    )
    b_d = tl.load(p_d)
    p_h = tl.make_block_ptr(
        base=h,
        shape=(BT, K, V),
        strides=(stride_b, stride_k, stride_v),
        offsets=(I_bh * BT, I_k[:, None], I_v[None, :]),
        block_shape=(BT, BK, BV),
        order=(1, 0, 2),
    )
    if USE_INITIAL_STATE:
        b_h = tl.load(p_initial_state)
    if NT > 1:
        b_sums = tl.zeros((BK, BV), dtype=tl.float32)
        tl.static_for(0, NT, lambda i: tl.multiple_of(b_h, V))
        for i in range(NT):
            b_sums += tl.dot(b_d, tl.load(p_k), allow_tf32=False)
            tl.store(p_h, b_h)
            b_h *= tl.sigmoid(b_h)
            if i < NT - 1:
                p_d.offsets += BT * stride_d
                p_k.offsets += BT * stride_k * stride_v
                p_h.offsets += BT * stride_k * stride_v
                b_d = tl.load(p_d)
    else:
        b_sums = tl.dot(b_d, tl.load(p_k), allow_tf32=False)
        tl.store(p_h, b_h)
    b_h_cumsum = tl.cumsum(b_sums, axis=0)
    p_v = tl.make_block_ptr(
        base=v,
        shape=(BT, K, V),
        strides=(stride_b, stride_k, stride_v),
        offsets=(I_bh * BT, i_k[:, None], i_v[None, :]),
        block_shape=(BT, BK, BV),
        order=(1, 0, 2),
    )
    b_v = tl.load(p_v)
    if STORE_FINAL_STATE:
        tl.store(p_final_state, b_h)
    b_h = b_h_cumsum[-1, :]
    b_o = b_v + b_h[:, None]
    p_v_new = tl.make_block_ptr(
        base=v_new,
        shape=(BT, K, V),
        strides=(stride_b, stride_k, stride_v),
        offsets=(I_bh * BT, i_k[:, None], i_v[None, :]),
        block_shape=(BT, BK, BV),
        order=(1, 0, 2),
    )
    tl.store(p_v_new, b_o)

def chunk_fwd_h_fn(
    k: Tensor,
    v: Tensor,
    d: Tensor,
    initial_state: Optional[Tensor] = None,
    final_state: Optional[Tensor] = None,
    max_size: int = 1024,
    max_len: Optional[int] = None,
    store_final_state: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    BT, BK, BV = d.shape[0], k.shape[1], v.shape[2]
    N, K, V = v.shape[0], k.shape[1], v.shape[2]
    NT = d.numel(0) // (K * v.numel(1))
    NB = max(1, triton.next_power_of_2(max_size // (4 * BK * BV)))
    BK = min(BK, max_size // (4
