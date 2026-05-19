import torch
import triton
import triton.language as tl

# Triton kernel for quantizing and copying KV tensor data
@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, DestLoc, Out, Out_scale,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr, BLOCK_GROUP_DIM: tl.constexpr,
):
    seq_len = tl.num_programs(0)
    head_num = tl.num_programs(1)
    head_dim = tl.num_programs(2)

    assert head_dim % BLOCK_GROUP_DIM == 0

    group_dim = head_dim // BLOCK_GROUP_DIM

    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_head_dim_offset = cur_head * head_dim

    dest_index = tl.load(DestLoc + cur_seq * head_num + cur_head)

    cur_group = (cur_head // BLOCK_GROUP_NUM) * group_size
    cur_group_dim = (cur_head % BLOCK_GROUP_NUM) * BLOCK_GROUP_DIM

    col_offsets = tl.arange(0, BLOCK_GROUP_DIM)
    mask = col_offsets < head_dim

    # step 1
    source_data = tl.load(K + cur_seq * head_dim * head_num + cur_head_dim_offset + col_offsets, mask=mask)

    abs_data = tl.abs(source_data)
    # step 2
    max_value = tl.max(tl.where(abs_data != 0, abs_data, 1e8), axis=0)

    # step 3
    source_data = source_data / max_value
    source_data = source_data.to(tl.int8)
    source_data = source_data + 127

    # step 4
    tl.store(Out + dest_index * group_dim + cur_group * BLOCK_GROUP_DIM + cur_group_dim + col_offsets, source_data, mask=mask)

    # step 5
    tl.store(Out_scale + dest_index * group_dim + cur_group * BLOCK_GROUP_DIM + cur_group_dim + col_offsets, max_value, mask=mask)

# Function to call the Triton kernel
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, group_size):
    assert K.shape[2] % 8 == 0
    assert K.shape[2] % 4 == 0 or torch.cuda.get_device_capability() >= (7, 0)

    BLOCK_GROUP_NUM = 4 if K.shape[2] % 16 == 0 else 2
    BLOCK_GROUP_DIM = K.shape[2] // BLOCK_GROUP_NUM

    kv_dim = K.shape[2]

    grid = (K.shape[0], K.shape[1])

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale, group_size,
        BLOCK_GROUP_NUM=BLOCK_GROUP_NUM,
        BLOCK_GROUP_DIM=BLOCK_GROUP_DIM,
        num_programs=(K.shape[0], K.shape[1], kv_dim),
        num_stages=1,
        grid=(K.shape[0], K.shape[1]),
    )
