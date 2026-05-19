import torch
import triton
import triton.language as tl

@triton.jit
def rwkv6_fused_forward_kernel(
    r_ptr, k_ptr, v_ptr, w_ptr, u_ptr, o_ptr,
    initial_state_ptr, final_state_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr
):
    # Calculate indices for parallel execution
    i_bh = tl.program_id(0)
    i_h = i_bh % H
    i_b = i_bh // H

    # Calculate pointers to the input data
    r_offset = i_b * H * T * K + i_h * T * K
    k_offset = i_b * H * T * K + i_h * T * K
    v_offset = i_b * H * T * V + i_h * T * V
    w_offset = i_b * H * T * K + i_h * T * K
    u_offset = i_h * K

    # Initialize state
    state = tl.zeros([K, V], dtype=tl.float32)
    if USE_INITIAL_STATE:
        initial_state_offset = i_b * H * K * V + i_h * K * V
        state += tl.load(initial_state_ptr + initial_state_offset)

    # Iterate over time steps
    for t in range(T):
        t_idx = T - 1 - t if REVERSE else t
        r_t = tl.load(r_ptr + r_offset + t_idx * K)
        k_t = tl.load(k_ptr + k_offset + t_idx * K)
        v_t = tl.load(v_ptr + v_offset + t_idx * V)
        w_t = tl.load(w_ptr + w_offset + t_idx * K)
        u_t = tl.load(u_ptr + u_offset)

        # Compute fused operation
        q_t = r_t * tl.exp(w_t)
        kv_t = tl.dot(k_t, v_t)
        o_t = tl.dot(state + kv_t * u_t, q_t)

        # Update state
        state = state * tl.exp(w_t) + kv_t

        # Store output
        o_offset = i_b * H * T * V + i_h * T * V + t_idx * V
        tl.store(o_ptr + o_offset, o_t)

    # Store final state if needed
    if STORE_FINAL_STATE:
        final_state_offset = i_b * H * K * V + i_h * K * V
        tl.store(final_state_ptr + final_state_offset, state)

# Wrapper function
def fused_recurrent_rwkv6(
    r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor,
    initial_state: torch.Tensor = None, output_final_state: bool = False, reverse: bool = False
):
    B, H, T, K = r.shape
    V = v.shape[-1]

    # Allocate output tensors
    o = torch.empty_like(v)
    final_state = torch.empty(B, H, K, V, dtype=torch.float32) if output_final_state else None

    # Launch Triton kernel
    grid = (B * H,)
    rwkv6_fused_forward_kernel[grid](
        r, k, v, w, u, o, initial_state, final_state,
        B=B, H=H, T=T, K=K, V=V,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        REVERSE=reverse
    )

    return o, final_state
