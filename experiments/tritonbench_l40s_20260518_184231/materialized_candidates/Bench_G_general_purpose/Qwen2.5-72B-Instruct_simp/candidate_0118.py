import triton
import triton.language as tl

@triton.jit
def fused_recurrent_rwkv6_kernel(
    r_ptr, k_ptr, v_ptr, w_ptr, u_ptr, o_ptr, state_ptr, scale, initial_state_ptr, output_final_state,
    BLOCK_SIZE: tl.constexpr, SEQ_LEN: tl.constexpr, HIDDEN_DIM: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load initial state if provided
    if initial_state_ptr is not None:
        state = tl.load(initial_state_ptr + offsets)
    else:
        state = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(SEQ_LEN):
        # Load input tensors for the current time step
        r = tl.load(r_ptr + t * HIDDEN_DIM + offsets)
        k = tl.load(k_ptr + t * HIDDEN_DIM + offsets)
        v = tl.load(v_ptr + t * HIDDEN_DIM + offsets)
        w = tl.load(w_ptr + t * HIDDEN_DIM + offsets)
        u = tl.load(u_ptr + t * HIDDEN_DIM + offsets)

        # Compute the recurrent state
        state = state * w + r * k
        o = state * u + v

        # Apply scaling if provided
        if scale is not None:
            o = o * scale

        # Store the output
        tl.store(o_ptr + t * HIDDEN_DIM + offsets, o)

    # Store the final state if required
    if output_final_state:
        tl.store(state_ptr + offsets, state)

import torch
import triton
import triton.language as tl

def fused_recurrent_rwkv6(r, k, v, w, u, scale=None, initial_state=None, output_final_state=False):
    # Get tensor dimensions
    SEQ_LEN, HIDDEN_DIM = r.shape

    # Allocate output tensor
    o = torch.empty_like(r)

    # Allocate state tensor if needed
    if output_final_state:
        final_state = torch.empty((HIDDEN_DIM,), device=r.device, dtype=r.dtype)
    else:
        final_state = None

    # Define grid and block dimensions
    BLOCK_SIZE = 256
    grid = (HIDDEN_DIM // BLOCK_SIZE,)

    # Launch the kernel
    fused_recurrent_rwkv6_kernel[grid](
        r, k, v, w, u, o, final_state, scale, initial_state, output_final_state,
        BLOCK_SIZE=BLOCK_SIZE, SEQ_LEN=SEQ_LEN, HIDDEN_DIM=HIDDEN_DIM
    )

    # Return the output and final state if required
    if output_final_state:
        return o, final_state
    else:
        return o
