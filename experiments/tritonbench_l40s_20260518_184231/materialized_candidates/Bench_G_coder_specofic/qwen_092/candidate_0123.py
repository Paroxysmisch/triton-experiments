import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr,  # input tensor
    z_ptr,  # output tensor
    m_ptr,  # lower triangular mask
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # number of heads
    T: tl.constexpr,  # sequence length
    S: tl.constexpr,  # feature size
    BS: tl.constexpr,  # spatial block size
    BT: tl.constexpr,  # time block size
    dtype: tl.constexpr,  # data type
):
    # Calculate block index
    block_idx = tl.program_id(0)
    b = block_idx // (H * (T // BT))
    h = (block_idx // (T // BT)) % H
    t_start = (block_idx % (T // BT)) * BT

    # Create block pointers
    s_block_ptr = tl.make_block_ptr(
        base_ptr=s_ptr,
        shape=(B, H, T, S),
        strides=(H * T * S, T * S, S, 1),
        offsets=(b * H * T * S + h * T * S + t_start * S, 0, 0, 0),
        block_shape=(BS, BS, BS, BS),
        order=(0, 1, 2, 3),
    )
    z_block_ptr = tl.make_block_ptr(
        base_ptr=z_ptr,
        shape=(B, H, T, S),
        strides=(H * T * S, T * S, S, 1),
        offsets=(b * H * T * S + h * T * S + t_start * S, 0, 0, 0),
        block_shape=(BS, BS, BS, BS),
        order=(0, 1, 2, 3),
    )
    m_block_ptr = tl.make_block_ptr(
        base_ptr=m_ptr,
        shape=(T, S),
        strides=(S, 1),
        offsets=(t_start * S, 0),
        block_shape=(BS, BS),
        order=(0, 1),
    )

    # Initialize cumulative sum
    b_z = tl.zeros((BS, BS, BS), dtype=dtype)

    # Iterate backwards over time blocks
    for t in range(tl.cdiv(T, BT) - 1, -1, -1):
        # Load input block
        b_s = tl.load(s_block_ptr, boundary_check=(True, True, True, True))

        # Load lower triangular mask
        m_s = tl.load(m_block_ptr, boundary_check=(True, True))

        # Compute masked dot product
        b_z = b_z + tl.dot(b_s, m_s, trans_b=True)

        # Store result in output block
        tl.store(z_block_ptr, b_z, boundary_check=(True, True, True, True))

        # Move to the next block
        s_block_ptr = tl.move(s_block_ptr, offset=(0, 0, 0, BS))
        z_block_ptr = tl.move(z_block_ptr, offset=(0, 0, 0, BS))
        m_block_ptr = tl.move(m_block_ptr, offset=(BS, 0))

    # Ensure boundary checks are done
    tl.block_barrier()
