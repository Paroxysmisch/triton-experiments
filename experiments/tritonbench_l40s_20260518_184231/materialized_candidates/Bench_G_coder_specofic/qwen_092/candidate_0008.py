import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    group_size, BLOCK_GROUP_NUM, BLOCK_GROUP_DIM,
    SEQUENCE_LEN, HEAD_COUNT, HEAD_DIM,
    BLOCK_SIZE=256
):
    # Define the block and grid size
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(HEAD_COUNT, BLOCK_GROUP_NUM * BLOCK_GROUP_DIM)
    num_groups = BLOCK_GROUP_NUM * BLOCK_GROUP_DIM
    group_id = pid % num_groups
    group_idx = pid // num_groups
    head_idx = group_idx // BLOCK_GROUP_DIM
    block_idx = group_idx % BLOCK_GROUP_DIM
    block_offset = block_idx * BLOCK_SIZE
    group_offset = group_id * group_size

    # Get the current head and sequence length
    head = head_idx * BLOCK_GROUP_DIM + block_idx
    seq_len = SEQUENCE_LEN

    # Define the block coordinates
    row = tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, group_size)

    # Compute the base offset for K
    base_offset = head * HEAD_DIM * seq_len + group_offset

    # Load data from K
    k_values = tl.load(K + base_offset + row[:, None] * HEAD_DIM + col[None, :])

    # Compute the scaling factor for the group
    max_abs_values = tl.reduce(tl.abs(k_values), col, tl.maximum)
    scaling_factors = max_abs_values / 127.0  # Assuming int8 range is -128 to 127

    # Quantize the data
    quantized_values = tl.round(k_values / scaling_factors[:, None])

    # Store the quantized values to Out
    dest_index = tl.load(Dest_loc + head * seq_len + row)
    out_offset = head * seq_len * 256 + dest_index * 256 + row
    tl.store(Out + out_offset, quantized_values)

    # Store the scaling factors to Out_scale
    out_scale_offset = head * seq_len * 256 + dest_index * 256 + row
    tl.store(Out_scale + out_scale_offset, scaling_factors)
