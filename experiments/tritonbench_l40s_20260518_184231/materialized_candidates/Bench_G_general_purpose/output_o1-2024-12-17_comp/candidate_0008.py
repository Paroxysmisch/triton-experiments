import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, DestLoc_ptr, Out_ptr, OutScale_ptr,
    B, H, D,
    group_size,
    BLOCK_GROUP_NUM, BLOCK_GROUP_DIM,
    strideKB, strideKH, strideKD,
    strideDB, strideDN,
    strideOB, strideOH, strideOD,
    strideSB, strideSH, strideSG,
    **meta
):
    # Program IDs: each block handles one (batch, head) pair and a chunk of the sequence dimension
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Offsets for groups within the head dimension
    # GROUP_NUM is how many groups we process per block, each group has group_size elements
    group_num_start = tl.program_id(2) * BLOCK_GROUP_NUM
    groups_offset = group_num_start + tl.arange(0, BLOCK_GROUP_NUM)
    # We'll clamp to avoid going beyond total number of groups
    valid_groups = groups_offset < (D // group_size)

    # Within each group, we process BLOCK_GROUP_DIM (<= group_size) elements
    elem_offset = tl.arange(0, BLOCK_GROUP_DIM)
    group_mask = elem_offset < group_size

    # Load the destination index for this sequence element
    # We assume that the "sequence dimension" is the second dimension of DestLoc.
    # The third program ID is effectively the "seq_id". Each group block corresponds to one sequence index.
    seq_id = tl.program_id(2)
    dest_idx = tl.load(
        DestLoc_ptr + batch_id * strideDB + seq_id * strideDN,
        mask=seq_id < D,  # Not strictly necessary if seq_id < seqlen
        other=0
    )

    # Loop over each group in this block
    for g in range(BLOCK_GROUP_NUM):
        group_index = groups_offset + g
        if not valid_groups[g]:
            continue
        # Calculate the head dimension offset for this group
        head_dim_offset = group_index[g] * group_size

        # Build final offsets to load from K
        k_offset = (batch_id * strideKB) + (head_id * strideKH) + (head_dim_offset + elem_offset) * strideKD
        load_mask = group_mask & (elem_offset + head_dim_offset < D)

        # Load data from K
        k_data = tl.load(K_ptr + k_offset, mask=load_mask, other=0.0)

        # Compute absolute max in this group
        abs_vals = tl.abs(k_data)
        group_max = tl.max(abs_vals, axis=0)

        # Store group scale
        # scale = group_max / 127.0 to map [-group_max, group_max] -> [-127, 127]
        scale = group_max / 127.0
        tl.store(
            OutScale_ptr
            + (batch_id * strideSB)
            + (head_id * strideSH)
            + group_index[g] * strideSG,
            scale
        )

        # Avoid divide-by-zero
        scale = tl.where(scale == 0.0, 1e-8, scale)

        # Quantize
        # round(...) -> cast to int -> clamp to [-128, 127]
        scaled = k_data / scale
        rounded = tl.round(scaled)
        clipped = tl.maximum(tl.minimum(rounded, 127.0), -128.0)
        q_data = clipped.to(tl.int8)

        # Store to Out using the destination index
        out_offset = (
            (batch_id * strideOB)
            + (head_id * strideOH)
            + (dest_idx * group_size + elem_offset + group_index[g] * group_size) * strideOD
        )
        tl.store(Out_ptr + out_offset, q_data, mask=load_mask)


def destindex_copy_quantize_kv(K, DestLoc, Out, OutScale, group_size, BLOCK_GROUP_NUM=1, BLOCK_GROUP_DIM=32):
    """
    High-level python interface for the _fwd_kernel_destindex_copy_quantize_kv kernel.
    K:          [batch, head, head_dim] float32
    DestLoc:    [batch, seqlen] int32 (destination indices)
    Out:        output int8
    OutScale:   output float32 (stores scale per group)
    group_size: int, group dimension for quantization
    """
    batch, head, head_dim = K.shape
    assert head_dim % group_size == 0, "head_dim must be divisible by group_size."

    # Strides
    strideKB = K.stride(0)
    strideKH = K.stride(1)
    strideKD = K.stride(2)

    strideDB = DestLoc.stride(0)
    strideDN = DestLoc.stride(1)

    strideOB = Out.stride(0)
    strideOH = Out.stride(1)
    strideOD = Out.stride(2)

    strideSB = OutScale.stride(0)
    strideSH = OutScale.stride(1)
    strideSG = OutScale.stride(2)

    # Grid dimensions
    # grid(0) = batch dimension
    # grid(1) = head dimension
    # grid(2) = seqlen dimension, where each block also handles BLOCK_GROUP_NUM groups
    seqlen = DestLoc.shape[1]
    # number of groups along head_dim
    total_groups = head_dim // group_size

    grid = (batch, head, seqlen)

    # Launch kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, OutScale,
        batch, head, head_dim,
        group_size,
        BLOCK_GROUP_NUM, BLOCK_GROUP_DIM,
        strideKB, strideKH, strideKD,
        strideDB, strideDN,
        strideOB, strideOH, strideOD,
        strideSB, strideSH, strideSG
    )
