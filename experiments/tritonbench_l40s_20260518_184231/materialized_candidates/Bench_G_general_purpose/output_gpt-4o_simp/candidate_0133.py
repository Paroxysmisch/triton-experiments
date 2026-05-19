import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(K, Dest_loc, Out, seq_len, head_num, stride_k_head, stride_k_seq, stride_out_head, stride_out_seq, BLOCK_SIZE: tl.constexpr):
    # Define the block indices
    head_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Compute the start position for this block
    start_idx = block_idx * BLOCK_SIZE

    # Loop over the sequence length
    for i in range(seq_len):
        # Calculate the index for this position
        dest_idx = tl.load(Dest_loc + i)

        # Compute the offset in the K tensor
        k_offset = head_idx * stride_k_head + start_idx + i * stride_k_seq
        # Compute the offset in the Out tensor
        out_offset = head_idx * stride_out_head + dest_idx * stride_out_seq

        # Load the value from K
        value = tl.load(K + k_offset)

        # Store the value in the Out tensor
        tl.store(Out + out_offset, value)


import torch

def destindex_copy_kv(K, Dest_loc, head_num, BLOCK_SIZE=128):
    # Determine the dimensions
    batch_size, _, seq_len = K.shape
    assert Dest_loc.shape[0] == seq_len, "Dest_loc length must match sequence length of K"

    # Prepare the output tensor
    Out = torch.empty_like(K)

    # Compute strides
    stride_k_head = K.stride(1)
    stride_k_seq = K.stride(2)
    stride_out_head = Out.stride(1)
    stride_out_seq = Out.stride(2)

    # Launch the Triton kernel
    grid = (head_num, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _fwd_kernel_destindex_copy_kv[grid](
        K, Dest_loc, Out, seq_len, head_num,
        stride_k_head, stride_k_seq, stride_out_head, stride_out_seq,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return Out
