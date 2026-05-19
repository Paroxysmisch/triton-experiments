import triton
import triton.language as tl

@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, k_batch_stride, k_head_stride, k_seq_stride, k_hidden_stride,
    v_ptr, v_batch_stride, v_head_stride, v_seq_stride, v_hidden_stride,
    d_ptr, d_batch_stride, d_head_stride, d_seq_stride, d_hidden_stride,
    v_new_ptr, v_new_batch_stride, v_new_head_stride, v_new_seq_stride, v_new_hidden_stride,
    h_ptr, h_batch_stride, h_head_stride, h_seq_stride, h_hidden_stride,
    initial_state_ptr, initial_state_batch_stride, initial_state_head_stride, initial_state_hidden_stride,
    final_state_ptr, final_state_batch_stride, final_state_head_stride, final_state_hidden_stride,
    NT: tl.constexpr, H: tl.constexpr, B: tl.constexpr, C: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    IS_INITIAL_STATE: tl.constexpr, IS_FINAL_STATE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (H * C)
    hid = (pid % (H * C)) // C
    cid = (pid % (H * C)) % C

    # Initialize block pointers
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr, shape=(B, H, NT, C), strides=(k_batch_stride, k_head_stride, k_seq_stride, k_hidden_stride),
        offsets=(bid, hid, 0, cid), block_shape=(1, 1, NT, 1), order=(0, 1, 2, 3)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr, shape=(B, H, NT, C), strides=(v_batch_stride, v_head_stride, v_seq_stride, v_hidden_stride),
        offsets=(bid, hid, 0, cid), block_shape=(1, 1, NT, 1), order=(0, 1, 2, 3)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr, shape=(B, H, NT, C), strides=(d_batch_stride, d_head_stride, d_seq_stride, d_hidden_stride),
        offsets=(bid, hid, 0, cid), block_shape=(1, 1, NT, 1), order=(0, 1, 2, 3)
    )
    v_new_block_ptr = tl.make_block_ptr(
        base=v_new_ptr, shape=(B, H, NT, C), strides=(v_new_batch_stride, v_new_head_stride, v_new_seq_stride, v_new_hidden_stride),
        offsets=(bid, hid, 0, cid), block_shape=(1, 1, NT, 1), order=(0, 1, 2, 3)
    )
    h_block_ptr = tl.make_block_ptr(
        base=h_ptr, shape=(B, H, NT, C), strides=(h_batch_stride, h_head_stride, h_seq_stride, h_hidden_stride),
        offsets=(bid, hid, 0, cid), block_shape=(1, 1, NT, 1), order=(0, 1, 2, 3)
    )

    if IS_INITIAL_STATE:
        initial_state_block_ptr = tl.make_block_ptr(
            base=initial_state_ptr, shape=(B, H, C), strides=(initial_state_batch_stride, initial_state_head_stride, initial_state_hidden_stride),
            offsets=(bid, hid, cid), block_shape=(1, 1, 1), order=(0, 1, 2)
        )

    if IS_FINAL_STATE:
        final_state_block_ptr = tl.make_block_ptr(
            base=final_state_ptr, shape=(B, H, C), strides=(final_state_batch_stride, final_state_head_stride, final_state_hidden_stride),
            offsets=(bid, hid, cid), block_shape=(1, 1, 1), order=(0, 1, 2)
        )

    # Initialize h with initial state if provided
    if IS_INITIAL_STATE:
        h = tl.load(initial_state_block_ptr)
    else:
        h = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(NT):
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        d = tl.load(d_block_ptr)

        # Compute new h
        h += k * v

        # Compute new v
        v_new = h * d

        # Store new v
        tl.store(v_new_block_ptr, v_new)

        # Update block pointers
        k_block_ptr = tl.advance(k_block_ptr, [0, 0, 1, 0])
        v_block_ptr = tl.advance(v_block_ptr, [0, 0, 1, 0])
        d_block_ptr = tl.advance(d_block_ptr, [0, 0, 1, 0])
        v_new_block_ptr = tl.advance(v_new_block_ptr, [0, 0, 1, 0])
        h_block_ptr = tl.advance(h_block_ptr, [0, 0, 1, 0])

    # Store final state if provided
    if IS_FINAL_STATE:
        tl.store(final_state_block_ptr, h)

import torch
import triton
import triton.language as tl

def chunk_fwd_h_fn(
    k: torch.Tensor,
    v: torch.Tensor,
    d: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    NT: int = 128,
    H: int = 16,
    B: int = 32,
    C: int = 64,
    BLOCK_SIZE: int = 128,
    IS_INITIAL_STATE: bool = False,
    IS_FINAL_STATE: bool = False
):
    # Initialize output tensors
    v_new = torch.zeros_like(v) if v_new is None else v_new
    h = torch.zeros((B, H, NT, C), dtype=torch.float32, device=k.device) if h is None else h

    # Calculate grid and block sizes
    grid = (B * H * C,)

    # Launch the kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        d, d.stride(0), d.stride(1), d.stride(2), d.stride(3),
        v_new, v_new.stride(0), v_new.stride(1), v_new.stride(2), v_new.stride(3),
        h, h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        initial_state, initial_state.stride(0), initial_state.stride(1), initial_state.stride(2) if initial_state is not None else 0,
        final_state, final_state.stride(0), final_state.stride(1), final_state.stride(2) if final_state is not None else 0,
        NT, H, B, C, BLOCK_SIZE,
        IS_INITIAL_STATE, IS_FINAL_STATE
    )

    return v_new, h
