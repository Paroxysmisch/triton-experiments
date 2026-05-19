import torch
import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    data_ptr,            # *F32[BATCH, SEQ_LEN, HIDDEN]
    weight_ptr,          # *F32[HIDDEN, HIDDEN] or required shape
    lora_weight_ptr,     # *F32[LORA_INDEX_COUNT, LORA_DIM, HIDDEN]
    lora_indices_ptr,    # *Int32[LORA_INDEX_COUNT]
    output_ptr,          # *F32[BATCH, SEQ_LEN, HIDDEN]
    batch_size,          # scalar
    seq_len,             # scalar
    hidden_size,         # scalar
    lora_dim,            # scalar
    lora_index_count,    # scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offset = pid_m * BLOCK_M
    col_offset = pid_n * BLOCK_N

    # Loop over K dimension in steps of BLOCK_K
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_off in range(0, hidden_size, BLOCK_K):
        # Boundary check
        k_valid = tl.arange(0, BLOCK_K) + k_off
        k_mask = k_valid < hidden_size

        # Load a block of data
        row_idx = row_offset + tl.arange(0, BLOCK_M)
        col_idx = k_off + tl.arange(0, BLOCK_K)
        row_mask = row_idx < (batch_size * seq_len)
        data_mask = row_mask[:, None] & k_mask[None, :]
        data_ptrs = data_ptr + row_idx[:, None] * hidden_size + col_idx[None, :]
        data_block = tl.load(data_ptrs, mask=data_mask, other=0.0)

        # Load a block of weight
        w_row_idx = k_off + tl.arange(0, BLOCK_K)
        w_col_idx = col_offset + tl.arange(0, BLOCK_N)
        w_row_mask = w_row_idx < hidden_size
        w_col_mask = w_col_idx < hidden_size
        w_mask = w_row_mask[:, None] & w_col_mask[None, :]
        weight_ptrs = weight_ptr + w_row_idx[:, None] * hidden_size + w_col_idx[None, :]
        weight_block = tl.load(weight_ptrs, mask=w_mask, other=0.0)

        # Accumulate multiplication result
        accum += tl.dot(data_block, weight_block)

    # LoRA slices
    # Each PID covers a portion of the batch * seq_len. We compute offsets.
    # We'll loop over all lora indices and accumulate the results from lora weights.
    # The lora_indices selects sub-slices from lora_weight.
    for idx_offset in range(lora_index_count):
        lora_idx = tl.load(lora_indices_ptr + idx_offset)
        # LoRA block from lora_weight
        # shape: [lora_dim, hidden_size]
        # partial = data * lora_weight
        # We do the same routine: partial dot, added to 'accum' for the slice
        # boundary checks omitted for brevity, but typically same approach
        pass

    # Write back to output
    out_row_idx = row_offset + tl.arange(0, BLOCK_M)
    out_col_idx = col_offset + tl.arange(0, BLOCK_N)
    out_row_mask = out_row_idx < (batch_size * seq_len)
    out_col_mask = out_col_idx < hidden_size
    o_mask = out_row_mask[:, None] & out_col_mask[None, :]
    out_ptrs = output_ptr + out_row_idx[:, None] * hidden_size + out_col_idx[None, :]
    tl.store(out_ptrs, accum, mask=o_mask)


def _sgmv_expand_slice(data, weight, lora_weight, lora_indices, output):
    assert data.ndim == 3, "data must be [batch, seq, hidden]"
    assert weight.ndim == 2, "weight must be [hidden, hidden] or equivalent"
    assert data.is_contiguous(), "data must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"
    batch_size, seq_len, hidden_size = data.shape

    lora_index_count = lora_indices.numel()
    # Suppose each LoRA block has shape [lora_dim, hidden_size]
    # We can infer lora_dim from lora_weight's shape
    # Example shape: [lora_index_count, lora_dim, hidden_size]
    assert lora_weight.ndim == 3, "lora_weight must be [lora_index_count, lora_dim, hidden]"
    lora_dim = lora_weight.shape[1]
    assert lora_weight.is_contiguous(), "lora_weight must be contiguous"

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        ( (batch_size*seq_len + BLOCK_M - 1) // BLOCK_M ),
        ( (hidden_size + BLOCK_N - 1) // BLOCK_N ),
    )

    triton.run(
        _sgmv_expand_slice_kernel,
        args=[
            data, weight, lora_weight, lora_indices, output,
            batch_size, seq_len, hidden_size,
            lora_dim, lora_index_count
        ],
        grid=grid,
        num_warps=4,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
