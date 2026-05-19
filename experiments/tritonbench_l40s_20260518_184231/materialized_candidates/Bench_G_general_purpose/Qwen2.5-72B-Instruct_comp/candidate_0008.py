import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,  # Source tensor of shape (batch, head, head_dim)
    DestLoc,  # Destination indices tensor
    Out,  # Output tensor to store quantized data
    Out_scale,  # Tensor to store scaling factors
    batch_size: tl.constexpr,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_groups = (head_dim + group_size - 1) // group_size
    group_id = pid % num_groups
    head_id = (pid // num_groups) % head_count
    batch_id = pid // (num_groups * head_count)

    group_start = group_id * group_size
    group_end = min(group_start + group_size, head_dim)

    # Compute the absolute maximum value for this group
    max_val = tl.zeros([BLOCK_GROUP_NUM], dtype=tl.float32)
    for i in range(group_start, group_end, BLOCK_GROUP_DIM):
        offsets = batch_id * head_count * head_dim + head_id * head_dim + i + tl.arange(0, BLOCK_GROUP_DIM)
        k_vals = tl.load(K + offsets, mask=offsets < (batch_id * head_count * head_dim + head_id * head_dim + group_end), other=0.0)
        max_val = tl.max(max_val, tl.abs(k_vals))

    # Reduce the max value across the block
    max_val = tl.reduce(max_val, axis=0, op='max')

    # Store the scaling factor
    scale = 127.0 / max_val
    scale_offset = batch_id * head_count * num_groups + head_id * num_groups + group_id
    tl.store(Out_scale + scale_offset, scale)

    # Quantize and store the data
    for i in range(group_start, group_end, BLOCK_GROUP_DIM):
        offsets = batch_id * head_count * head_dim + head_id * head_dim + i + tl.arange(0, BLOCK_GROUP_DIM)
        k_vals = tl.load(K + offsets, mask=offsets < (batch_id * head_count * head_dim + head_id * head_dim + group_end), other=0.0)
        quantized_vals = tl.cast(tl.round(k_vals * scale), tl.int8)
        dest_indices = tl.load(DestLoc + offsets, mask=offsets < (batch_id * head_count * head_dim + head_id * head_dim + group_end), other=0)
        out_offsets = batch_id * head_count * head_dim + head_id * head_dim + dest_indices
        tl.store(Out + out_offsets, quantized_vals, mask=offsets < (batch_id * head_count * head_dim + head_id * head_dim + group_end))

import torch

def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, group_size, BLOCK_GROUP_NUM, BLOCK_GROUP_DIM):
    batch_size, head_count, head_dim = K.shape
    num_groups = (head_dim + group_size - 1) // group_size

    # Ensure head_dim is divisible by group_size
    assert head_dim % group_size == 0, "head_dim must be divisible by group_size"

    # Set up the grid
    grid = (batch_size * head_count * num_groups,)

    # Launch the kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        batch_size, head_count, head_dim, group_size,
        BLOCK_GROUP_NUM, BLOCK_GROUP_DIM
    )

# Example usage
batch_size = 2
head_count = 4
head_dim = 128
group_size = 32
BLOCK_GROUP_NUM = 1
BLOCK_GROUP_DIM = 32

K = torch.randn((batch_size, head_count, head_dim), device='cuda')
DestLoc = torch.randint(0, head_dim, (batch_size, head_count, head_dim), device='cuda')
Out = torch.zeros((batch_size, head_count, head_dim), dtype=torch.int8, device='cuda')
Out_scale = torch.zeros((batch_size, head_count, (head_dim + group_size - 1) // group_size), dtype=torch.float32, device='cuda')

destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, group_size, BLOCK_GROUP_NUM, BLOCK_GROUP_DIM)
