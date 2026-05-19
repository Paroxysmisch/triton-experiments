import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config(meta={'BLOCK_K': 64, 'BLOCK_V': 64, 'num_warps': 1}),
        triton.Config(meta={'BLOCK_K': 64, 'BLOCK_V': 64, 'num_warps': 2}),
        triton.Config(meta={'BLOCK_K': 64, 'BLOCK_V': 64, 'num_warps': 4}),
        triton.Config(meta={'BLOCK_K': 64, 'BLOCK_V': 64, 'num_warps': 8}),
        triton.Config(meta={'BLOCK_K': 64, 'BLOCK_V': 64, 'num_warps': 16}),
        triton.Config(meta={'BLOCK_K': 64, 'BLOCK_V': 64, 'num_warps': 32}),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, h_ptr, vnew_ptr,
    initial_state_ptr, final_state_ptr,
    BT, BK, BV, NT,
    stride_kb, stride_kt, stride_kv,
    stride_vb, stride_vt, stride_vv,
    stride_db, stride_dt, stride_dv,
    stride_hb, stride_ht, stride_hv,
    stride_vnewb, stride_vnewt, stride_vnewv,
    stride_isb, stride_isv,
    stride_fsb, stride_fsv,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_k  = tl.program_id(1)
    i_v  = tl.program_id(2)

    b_idx = i_bh // BT
    t_idx = i_bh % BT

    # Pointers to K, V, D, H, V_new based on the block indices
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + b_idx * stride_kb + t_idx * stride_kt + i_k * BLOCK_K * stride_kv,
        shape=(BK, BV),
        strides=(stride_kv, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, 1),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + b_idx * stride_vb + t_idx * stride_vt + i_v * BLOCK_V * stride_vv,
        shape=(BK, BV),
        strides=(stride_vv, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_V, 1),
        order=(0, 1),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + b_idx * stride_db + t_idx * stride_dt,
        shape=(BK, BV),
        strides=(stride_dv, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_V),
        order=(0, 1),
    )
    h_block_ptr = tl.make_block_ptr(
        base=h_ptr + b_idx * stride_hb + t_idx * stride_ht,
        shape=(BK, BV),
        strides=(stride_hv, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_V),
        order=(0, 1),
    )
    vnew_block_ptr = tl.make_block_ptr(
        base=vnew_ptr + b_idx * stride_vnewb + t_idx * stride_vnewt,
        shape=(BK, BV),
        strides=(stride_vnewv, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_V),
        order=(0, 1),
    )

    # Load initial state if needed
    b_h = tl.zeros((BLOCK_V,), dtype=tl.float32)
    if USE_INITIAL_STATE:
        init_state_ptr = tl.make_block_ptr(
            base=initial_state_ptr + b_idx * stride_isb,
            shape=(BV,),
            strides=(stride_isv,),
            offsets=(0,),
            block_shape=(BLOCK_V,),
            order=(0,),
        )
        b_h = tl.load(init_state_ptr)

    # Iterate over NT in time dimension
    for nt in range(NT):
        # Compute offsets in K, V, D for the current time step
        k_t_block_ptr = k_block_ptr + nt * BLOCK_K * stride_kt
        v_t_block_ptr = v_block_ptr + nt * BLOCK_V * stride_vt
        d_t_block_ptr = d_block_ptr + nt * BLOCK_V * stride_dt
        h_t_block_ptr = h_block_ptr + nt * BLOCK_V * stride_ht
        vnew_t_block_ptr = vnew_block_ptr + nt * BLOCK_V * stride_vnewt

        # Load K, V, D
        k_val = tl.load(k_t_block_ptr)
        v_val = tl.load(v_t_block_ptr)
        d_val = tl.load(d_t_block_ptr)

        # Save current state to H
        tl.store(h_t_block_ptr, b_h.broadcast((BLOCK_K, BLOCK_V)))

        # Perform the update for V using K and D
        kv_mul = tl.dot(k_val.to(tl.float32), d_val.to(tl.float32))
        v_updated = v_val.to(tl.float32) + kv_mul
        tl.store(vnew_t_block_ptr, v_updated)

        # Accumulate the result to b_h
        b_h_cumsum = b_h + tl.sum(kv_mul, axis=0)
        b_h = b_h_cumsum

    # Store final state if needed
    if STORE_FINAL_STATE:
        final_state_block_ptr = tl.make_block_ptr(
            base=final_state_ptr + b_idx * stride_fsb,
            shape=(BV,),
            strides=(stride_fsv,),
            offsets=(0,),
            block_shape=(BLOCK_V,),
            order=(0,),
        )
        tl.store(final_state_block_ptr, b_h)


def chunk_fwd_h_fn(
    k: torch.Tensor,
    v: torch.Tensor,
    d: torch.Tensor,
    NT: int,
    use_initial_state: bool = False,
    store_final_state: bool = False,
    initial_state: torch.Tensor = None,
    final_state: torch.Tensor = None
):
    BT, BK, BV = k.size(0), k.size(-2), v.size(-1)
    grid_bh = BT * NT
    grid_k = (BK + 63) // 64
    grid_v = (BV + 63) // 64

    # Allocate output
    h = torch.zeros_like(d, dtype=torch.float32)
    v_new = torch.zeros_like(v, dtype=torch.float32)

    # Strides
    stride_kb = k.stride(0)
    stride_kt = k.stride(1)
    stride_kv = k.stride(2) if k.ndim == 3 else 1
    stride_vb = v.stride(0)
    stride_vt = v.stride(1)
    stride_vv = v.stride(2) if v.ndim == 3 else 1
    stride_db = d.stride(0)
    stride_dt = d.stride(1)
    stride_dv = d.stride(2) if d.ndim == 3 else 1
    stride_hb = h.stride(0)
    stride_ht = h.stride(1)
    stride_hv = h.stride(2) if h.ndim == 3 else 1
    stride_vnewb = v_new.stride(0)
    stride_vnewt = v_new.stride(1)
    stride_vnewv = v_new.stride(2) if v_new.ndim == 3 else 1

    stride_isb = 0
    stride_isv = 0
    if initial_state is not None:
        stride_isb = initial_state.stride(0)
        stride_isv = initial_state.stride(1) if initial_state.ndim == 2 else 1

    stride_fsb = 0
    stride_fsv = 0
    if final_state is not None:
        stride_fsb = final_state.stride(0)
        stride_fsv = final_state.stride(1) if final_state.ndim == 2 else 1

    chunk_delta_rule_fwd_kernel_h[
        (grid_bh, grid_k, grid_v)
    ](
        k, v, d, h, v_new,
        initial_state if initial_state is not None else torch.empty(0, device=k.device),
        final_state if final_state is not None else torch.empty(0, device=k.device),
        BT, BK, BV, NT,
        stride_kb, stride_kt, stride_kv,
        stride_vb, stride_vt, stride_vv,
        stride_db, stride_dt, stride_dv,
        stride_hb, stride_ht, stride_hv,
        stride_vnewb, stride_vnewt, stride_vnewv,
        stride_isb, stride_isv,
        stride_fsb, stride_fsv,
        use_initial_state,
        store_final_state
    )
    return h, v_new
