import triton
import triton.language as tl
import torch

@triton.jit
def _sgmv_expand_slice_kernel(
    data_ptr,                    # [batch, seq_len, hidden_dim]
    data_batch_stride,
    data_seq_stride,
    data_hidden_stride,
    lora_ptr,                    # [num_lora_indices, hidden_dim, r] or similar layout
    lora_hidden_stride,
    lora_r_stride,
    lora_indices_ptr,            # [num_lora_indices] -> indicates slices for LoRA
    out_ptr,                     # [batch, seq_len, hidden_dim]
    out_batch_stride,
    out_seq_stride,
    out_hidden_stride,
    num_lora_indices: tl.constexpr,
    BLOCK_M: tl.constexpr,       # typically covers the sequence dimension portion
    BLOCK_N: tl.constexpr,       # typically covers the hidden dimension portion
    BLOCK_K: tl.constexpr,       # typically covers LoRA "r" dimension or partial accum dims
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    # Offsets in sequence dimension and batch dimension
    off_m = pid_m * BLOCK_M
    off_b = pid_b

    # Pointer base offsets
    data_base = data_ptr + off_b * data_batch_stride + off_m * data_seq_stride
    out_base = out_ptr + off_b * out_batch_stride + off_m * out_seq_stride

    # Create a [BLOCK_M, BLOCK_N] accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over LoRA indices dimension (summing expansions)
    for idx_idx in range(num_lora_indices):
        # Read the index for which slice to use in the LoRA weights
        lora_idx = tl.load(lora_indices_ptr + idx_idx)
        # Each program accumulates partial expansions over 'r' dimension in smaller chunks
        # We break r dimension into BLOCK_K sized slices
        rk = 0
        while rk < BLOCK_K:
            # Offsets in LoRA weight
            # lora_ptr layout expected: [num_lora_indices, hidden_dim, r]
            # we gather each chunk in the r dimension
            off_r = rk
            if off_r >= BLOCK_K:
                break

            # Create blocking for M dimension in data
            m_range = tl.arange(0, BLOCK_M)
            # Create blocking for N dimension in hidden
            n_range = tl.arange(0, BLOCK_N)
            # Offsets compute
            # data: shape [BLOCK_M (seq), BLOCK_K/r ... (embedding)]
            # gather from data in hidden dimension = n_range
            # boundary checks
            seq_mask = (off_m + m_range) < (data_seq_stride // data_hidden_stride)
            n_mask = n_range < (data_hidden_stride // 1)
            mask = seq_mask[:, None] & n_mask[None, :]

            # Load data (batch, seq, hidden) for each (m,n)
            data_offset = data_base + m_range[:, None] * data_hidden_stride + n_range[None, :]
            data_val = tl.where(
                mask,
                tl.load(data_offset, mask=mask, other=0.0).to(tl.float32),
                0.0
            )

            # Load LoRA weight portion, shape [hidden_dim, r]
            # hidden offset is n_range, r offset is off_r
            lora_offset = (lora_ptr
                           + lora_idx * (lora_hidden_stride * BLOCK_N)
                           + n_range[:, None] * lora_hidden_stride
                           + (off_r + tl.arange(0, 1))[None, :])
            lora_val = tl.where(
                n_mask[:, None],
                tl.load(lora_offset, mask=n_mask[:, None], other=0.0).to(tl.float32),
                0.0
            )
            # Broadcast LoRA over M dimension
            lora_val_broadcast = tl.broadcast_to(lora_val, (BLOCK_M, BLOCK_N))

            # Outer product in 'r' direction
            # We assume each partial chunk in 'r' dimension contributes
            acc += data_val * lora_val_broadcast

            rk += 1  # move to next block of the r dimension

    # Store results
    # boundary checks
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    seq_mask = (off_m + m_range) < (data_seq_stride // data_hidden_stride)
    n_mask = n_range < (data_hidden_stride // 1)
    mask = seq_mask[:, None] & n_mask[None, :]

    out_offset = out_base + m_range[:, None] * out_hidden_stride + n_range[None, :]
    tl.store(out_offset, acc.to(tl.float16), mask=mask)


def _sgmv_expand_slice(
    data: torch.Tensor,
    lora_weights: torch.Tensor,
    lora_indices: torch.Tensor,
    out: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 8
):
    """
    Wrapper function for the _sgmv_expand_slice_kernel.
    data:           [batch, seq_len, hidden_dim] (float16 or float32)
    lora_weights:   [num_lora_indices, hidden_dim, r] (float16 or float32)
    lora_indices:   [num_lora_indices] (int32)
    out:            [batch, seq_len, hidden_dim] (float16)
    """

    # Basic checks
    assert data.is_cuda, "data must be on CUDA"
    assert lora_weights.is_cuda, "lora_weights must be on CUDA"
    assert lora_indices.is_cuda, "lora_indices must be on CUDA"
    assert out.is_cuda, "out must be on CUDA"
    assert data.dim() == 3, "data must be [batch, seq_len, hidden_dim]"
    assert out.dim() == 3, "out must be [batch, seq_len, hidden_dim]"
    bsz, seq_len, hidden_dim = data.shape
    num_lora_indices, lora_hidden_dim, _ = lora_weights.shape
    assert lora_hidden_dim == hidden_dim, "LoRA weights hidden dim mismatch"
    assert out.shape == (bsz, seq_len, hidden_dim), "out shape mismatch"

    data_ptr = data.data_ptr()
    lora_ptr = lora_weights.data_ptr()
    lora_indices_ptr = lora_indices.data_ptr()
    out_ptr = out.data_ptr()

    # Compute strides
    data_batch_stride = data.stride(0)
    data_seq_stride = data.stride(1)
    data_hidden_stride = data.stride(2)

    lora_hidden_stride = lora_weights.stride(1)
    lora_r_stride = lora_weights.stride(2)

    out_batch_stride = out.stride(0)
    out_seq_stride = out.stride(1)
    out_hidden_stride = out.stride(2)

    # Grid setup
    # Launch a 2D grid: (seq_blocks, batch)
    seq_blocks = (seq_len + block_m - 1) // block_m
    grid = (seq_blocks, bsz)

    triton.run(
        _sgmv_expand_slice_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        # kernel args
        data_ptr,
        data_batch_stride,
        data_seq_stride,
        data_hidden_stride,
        lora_ptr,
        lora_hidden_stride,
        lora_r_stride,
        lora_indices_ptr,
        out_ptr,
        out_batch_stride,
        out_seq_stride,
        out_hidden_stride,
        num_lora_indices,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k
    )
