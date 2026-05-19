import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=8),
        triton.Config({'BT': 128, 'BK': 64, 'BV': 64}, num_warps=4),
        triton.Config({'BT': 256, 'BK': 64, 'BV': 64}, num_warps=8),
        triton.Config({'BT': 64, 'BK': 128, 'BV': 64}, num_warps=8),
        triton.Config({'BT': 64, 'BK': 64, 'BV': 128}, num_warps=8),
    ],
    key=['K', 'V', 'BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    # Pointers to tensors
    k_ptr, v_ptr, d_ptr,
    h_initial_ptr, h_final_ptr,
    v_new_ptr,
    # Tensor dimensions
    B, H, T, K, V,
    # Strides for tensors
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    stride_v_b, stride_v_h, stride_v_t, stride_v_v,
    stride_d_b, stride_d_h, stride_d_t, stride_d_k, stride_d_v,
    stride_h_initial_b, stride_h_initial_h, stride_h_initial_k, stride_h_initial_v,
    stride_h_final_b, stride_h_final_h, stride_h_final_k, stride_h_final_v,
    stride_v_new_b, stride_v_new_h, stride_v_new_t, stride_v_new_k, stride_v_new_v,
    # Kernel parameters
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NT: tl.constexpr,
):
    # Program indices
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    b = i_bh // H
    h = i_bh % H

    # Offset calculations
    offs_k = i_k * BK + tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_t = tl.arange(0, BT)

    # Block pointers for K, V, D
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + b * stride_k_b + h * stride_k_h,
        shape=(T, K),
        strides=(stride_k_t, stride_k_k),
        offsets=(0, offs_k[0]),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + b * stride_v_b + h * stride_v_h,
        shape=(T, V),
        strides=(stride_v_t, stride_v_v),
        offsets=(0, offs_v[0]),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + b * stride_d_b + h * stride_d_h,
        shape=(T, K, V),
        strides=(stride_d_t, stride_d_k, stride_d_v),
        offsets=(0, offs_k[0], offs_v[0]),
        block_shape=(BT, BK, BV),
        order=(2, 1, 0)
    )

    # Initialize state
    if USE_INITIAL_STATE:
        h_ptr = h_initial_ptr + b * stride_h_initial_b + h * stride_h_initial_h
        h_block_ptr = tl.make_block_ptr(
            base=h_ptr,
            shape=(K, V),
            strides=(stride_h_initial_k, stride_h_initial_v),
            offsets=(offs_k[0], offs_v[0]),
            block_shape=(BK, BV),
            order=(1, 0)
        )
        b_h = tl.load(h_block_ptr, boundary_check=(0, 1))
    else:
        b_h = tl.zeros((BK, BV), dtype=tl.float32)

    # Temporal loop
    for t in range(NT):
        # Load current K, V, D blocks
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))
        d = tl.load(d_block_ptr, boundary_check=(0, 1, 2))
        
        # Compute delta update
        product = tl.dot(k, v, allow_tf32=False)
        product = product * d
        b_h += tl.sum(product, axis=0)

        # Update block pointers for next time step
        k_block_ptr = tl.advance(k_block_ptr, (BT, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BT, 0))
        d_block_ptr = tl.advance(d_block_ptr, (BT, 0, 0))

    # Store final state
    if STORE_FINAL_STATE:
        h_final_ptr = h_final_ptr + b * stride_h_final_b + h * stride_h_final_h
        h_final_block_ptr = tl.make_block_ptr(
            base=h_final_ptr,
            shape=(K, V),
            strides=(stride_h_final_k, stride_h_final_v),
            offsets=(offs_k[0], offs_v[0]),
            block_shape=(BK, BV),
            order=(1, 0)
        )
        tl.store(h_final_block_ptr, b_h, boundary_check=(0, 1))

    # Store result to v_new
    v_new_block_ptr = tl.make_block_ptr(
        base=v_new_ptr + b * stride_v_new_b + h * stride_v_new_h,
        shape=(K, V),
        strides=(stride_v_new_k, stride_v_new_v),
        offsets=(offs_k[0], offs_v[0]),
        block_shape=(BK, BV),
        order=(1, 0)
    )
    tl.store(v_new_block_ptr, b_h, boundary_check=(0, 1))

def chunk_fwd_h_fn(k: torch.Tensor, v: torch.Tensor, d: torch.Tensor, initial_state: torch.Tensor = None):
    B, H, T, K = k.shape
    _, _, _, V = v.shape
    device = k.device

    # Determine block sizes
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    BT = 64  # Starting value, autotune will adjust

    # Ensure block sizes are valid
    assert BK >= 16 and (BK & (BK-1)) == 0, "BK must be a power of two >=16"
    assert BV >= 16 and (BV & (BV-1)) == 0, "BV must be a power of two >=16"

    # Calculate number of time steps
    NT = T // BT
    if T % BT != 0:
        NT += 1  # Handle residual

    # Allocate output tensors
    v_new = torch.empty((B, H, K, V), device=device, dtype=k.dtype)
    h_final = torch.empty_like(v_new) if initial_state is not None else None

    # Grid configuration
    grid = (triton.cdiv(K, BK), triton.cdiv(V, BV), B * H)

    # Kernel parameters
    kernel_params = {
        'k_ptr': k, 'v_ptr': v, 'd_ptr': d,
        'h_initial_ptr': initial_state if initial_state is not None else torch.empty(0, device=device),
        'h_final_ptr': h_final if h_final is not None else torch.empty(0, device=device),
        'v_new_ptr': v_new,
        'B': B, 'H': H, 'T': T, 'K': K, 'V': V,
        'stride_k_b': k.stride(0), 'stride_k_h': k.stride(1), 'stride_k_t': k.stride(2), 'stride_k_k': k.stride(3),
        'stride_v_b': v.stride(0), 'stride_v_h': v.stride(1), 'stride_v_t': v.stride(2), 'stride_v_v': v.stride(3),
        'stride_d_b': d.stride(0), 'stride_d_h': d.stride(1), 'stride_d_t': d.stride(2), 'stride_d_k': d.stride(3), 'stride_d_v': d.stride(4),
        'stride_h_initial_b': initial_state.stride(0) if initial_state is not None else 0,
        'stride_h_initial_h': initial_state.stride(1) if initial_state is not None else 0,
        'stride_h_initial_k': initial_state.stride(2) if initial_state is not None else 0,
        'stride_h_initial_v': initial_state.stride(3) if initial_state is not None else 0,
        'stride_h_final_b': h_final.stride(0) if h_final is not None else 0,
        'stride_h_final_h': h_final.stride(1) if h_final is not None else 0,
        'stride_h_final_k': h_final.stride(2) if h_final is not None else 0,
        'stride_h_final_v': h_final.stride(3) if h_final is not None else 0,
        'stride_v_new_b': v_new.stride(0), 'stride_v_new_h': v_new.stride(1),
        'stride_v_new_k': v_new.stride(2), 'stride_v_new_v': v_new.stride(3),
        'USE_INITIAL_STATE': initial_state is not None,
        'STORE_FINAL_STATE': h_final is not None,
        'BT': BT, 'BK': BK, 'BV': BV, 'NT': NT
    }

    # Launch kernel
    chunk_delta_rule_fwd_kernel_h[grid](**kernel_params)
    return v_new, h_final
