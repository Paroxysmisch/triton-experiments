import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    BT, BK, BV,
    BLOCK_SIZE_BT: tl.constexpr, BLOCK_SIZE_BK: tl.constexpr, BLOCK_SIZE_BV: tl.constexpr,
    q_batch_stride, q_head_stride, q_seq_stride, q_hidden_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_hidden_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_hidden_stride,
    h_batch_stride, h_head_stride, h_seq_stride, h_hidden_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_hidden_stride,
    scale: tl.constexpr
):
    # Get the current block's coordinates in the grid
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    # Compute the starting indices for the current block
    q_offset = bid * q_batch_stride + hid * q_head_stride
    k_offset = bid * k_batch_stride + hid * k_head_stride
    v_offset = bid * v_batch_stride + hid * v_head_stride
    h_offset = bid * h_batch_stride + hid * h_head_stride
    o_offset = bid * o_batch_stride + hid * o_head_stride

    # Initialize the block pointers
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(BT, BK),
        strides=(q_seq_stride, q_hidden_stride),
        offsets=(pid * BLOCK_SIZE_BT, 0),
        block_shape=(BLOCK_SIZE_BT, BLOCK_SIZE_BK),
        order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(BK, BT),
        strides=(k_hidden_stride, k_seq_stride),
        offsets=(0, pid * BLOCK_SIZE_BT),
        block_shape=(BLOCK_SIZE_BK, BLOCK_SIZE_BT),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_offset,
        shape=(BT, BV),
        strides=(v_seq_stride, v_hidden_stride),
        offsets=(pid * BLOCK_SIZE_BT, 0),
        block_shape=(BLOCK_SIZE_BT, BLOCK_SIZE_BV),
        order=(1, 0)
    )
    h_block_ptr = tl.make_block_ptr(
        base=h_ptr + h_offset,
        shape=(BT, BK),
        strides=(h_seq_stride, h_hidden_stride),
        offsets=(pid * BLOCK_SIZE_BT, 0),
        block_shape=(BLOCK_SIZE_BT, BLOCK_SIZE_BK),
        order=(1, 0)
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(BT, BV),
        strides=(o_seq_stride, o_hidden_stride),
        offsets=(pid * BLOCK_SIZE_BT, 0),
        block_shape=(BLOCK_SIZE_BT, BLOCK_SIZE_BV),
        order=(1, 0)
    )

    # Load the query, key, and value blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # Compute the attention scores
    attn_scores = tl.dot(q, k, allow_tf32=True) * scale

    # Apply the auxiliary tensor (h) if provided
    if h_ptr is not None:
        h = tl.load(h_block_ptr)
        attn_scores += h

    # Apply the softmax function
    attn_scores = tl.softmax(attn_scores, axis=1)

    # Compute the output
    o = tl.dot(attn_scores, v, allow_tf32=True)

    # Store the output
    tl.store(o_block_ptr, o)

import torch

def chunk_fwd_o_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: Optional[torch.Tensor],
    o: torch.Tensor,
    BT: int,
    BK: int,
    BV: int,
    scale: float,
    BLOCK_SIZE_BT: int = 128,
    BLOCK_SIZE_BK: int = 128,
    BLOCK_SIZE_BV: int = 128,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
):
    # Get the grid dimensions
    grid = (q.shape[2] // BLOCK_SIZE_BT, q.shape[0], q.shape[1])

    # Ensure the grid dimensions are within the maximum grid size
    if max_grid is not None:
        grid = tuple(min(g, mg) for g, mg in zip(grid, max_grid))

    # Launch the kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, o,
        BT, BK, BV,
        BLOCK_SIZE_BT, BLOCK_SIZE_BK, BLOCK_SIZE_BV,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h.stride(0) if h is not None else 0, h.stride(1) if h is not None else 0, h.stride(2) if h is not None else 0, h.stride(3) if h is not None else 0,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale
    )
