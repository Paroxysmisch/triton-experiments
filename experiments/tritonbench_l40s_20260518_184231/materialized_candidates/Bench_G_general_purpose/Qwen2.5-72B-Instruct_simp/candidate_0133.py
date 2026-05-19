import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K_ptr,  # Pointer to the input tensor K
    Dest_loc_ptr,  # Pointer to the destination indices tensor
    Out_ptr,  # Pointer to the output tensor
    seq_len,  # Length of the sequence (length of Dest_loc)
    head_num,  # Number of heads
    K_batch_stride,  # Stride for batch dimension in K
    K_head_stride,  # Stride for head dimension in K
    K_seq_stride,  # Stride for sequence dimension in K
    K_hidden_stride,  # Stride for hidden dimension in K
    Dest_loc_batch_stride,  # Stride for batch dimension in Dest_loc
    Dest_loc_head_stride,  # Stride for head dimension in Dest_loc
    Dest_loc_seq_stride,  # Stride for sequence dimension in Dest_loc
    Out_batch_stride,  # Stride for batch dimension in Out
    Out_head_stride,  # Stride for head dimension in Out
    Out_seq_stride,  # Stride for sequence dimension in Out
    Out_hidden_stride,  # Stride for hidden dimension in Out
    BLOCK_SEQ: tl.constexpr,  # Block size for sequence dimension
    BLOCK_HEAD: tl.constexpr,  # Block size for head dimension
):
    # Get the current block's indices
    pid = tl.program_id(axis=0)
    num_blocks_seq = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    num_blocks_head = (head_num + BLOCK_HEAD - 1) // BLOCK_HEAD
    head_id = pid % num_blocks_head
    seq_id = pid // num_blocks_head

    # Compute the starting indices for the current block
    head_start = head_id * BLOCK_HEAD
    seq_start = seq_id * BLOCK_SEQ

    # Load the destination indices for the current block
    dest_indices = tl.load(Dest_loc_ptr + seq_start * Dest_loc_seq_stride + head_start * Dest_loc_head_stride + tl.arange(0, BLOCK_SEQ))

    # Iterate over the head and sequence dimensions
    for h in range(head_start, min(head_start + BLOCK_HEAD, head_num)):
        for s in range(seq_start, min(seq_start + BLOCK_SEQ, seq_len)):
            # Load the source data from K
            k_val = tl.load(K_ptr + s * K_seq_stride + h * K_head_stride + tl.arange(0, K_hidden_stride))

            # Compute the destination index
            dest_index = dest_indices[s - seq_start]

            # Store the data in the output tensor
            tl.store(Out_ptr + dest_index * Out_seq_stride + h * Out_head_stride + tl.arange(0, Out_hidden_stride), k_val)

import torch
import triton
import triton.language as tl

def destindex_copy_kv(K, Dest_loc, Out):
    # Check shape consistency
    assert K.shape == Out.shape, "Input tensor K and output tensor Out must have the same shape"
    assert Dest_loc.shape[0] == K.shape[1], "Length of Dest_loc must match the sequence length of K"

    # Get the dimensions
    batch_size, seq_len, head_num, hidden_dim = K.shape

    # Get the strides
    K_batch_stride = K.stride(0)
    K_head_stride = K.stride(2)
    K_seq_stride = K.stride(1)
    K_hidden_stride = K.stride(3)

    Dest_loc_batch_stride = Dest_loc.stride(0)
    Dest_loc_head_stride = Dest_loc.stride(1)
    Dest_loc_seq_stride = Dest_loc.stride(2)

    Out_batch_stride = Out.stride(0)
    Out_head_stride = Out.stride(2)
    Out_seq_stride = Out.stride(1)
    Out_hidden_stride = Out.stride(3)

    # Define block sizes
    BLOCK_SEQ = 128
    BLOCK_HEAD = 8

    # Compute grid size
    grid = (seq_len * head_num // (BLOCK_SEQ * BLOCK_HEAD),)

    # Launch the kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K,  # Pointer to the input tensor K
        Dest_loc,  # Pointer to the destination indices tensor
        Out,  # Pointer to the output tensor
        seq_len,  # Length of the sequence (length of Dest_loc)
        head_num,  # Number of heads
        K_batch_stride,  # Stride for batch dimension in K
        K_head_stride,  # Stride for head dimension in K
        K_seq_stride,  # Stride for sequence dimension in K
        K_hidden_stride,  # Stride for hidden dimension in K
        Dest_loc_batch_stride,  # Stride for batch dimension in Dest_loc
        Dest_loc_head_stride,  # Stride for head dimension in Dest_loc
        Dest_loc_seq_stride,  # Stride for sequence dimension in Dest_loc
        Out_batch_stride,  # Stride for batch dimension in Out
        Out_head_stride,  # Stride for head dimension in Out
        Out_seq_stride,  # Stride for sequence dimension in Out
        Out_hidden_stride,  # Stride for hidden dimension in Out
        BLOCK_SEQ,  # Block size for sequence dimension
        BLOCK_HEAD,  # Block size for head dimension
    )

# Example usage
if __name__ == "__main__":
    # Create example tensors
    batch_size = 2
    seq_len = 1024
    head_num = 16
    hidden_dim = 64

    K = torch.randn((batch_size, seq_len, head_num, hidden_dim), device="cuda")
    Dest_loc = torch.randint(0, seq_len, (batch_size, head_num, seq_len), device="cuda")
    Out = torch.zeros((batch_size, seq_len, head_num, hidden_dim), device="cuda")

    # Call the wrapper function
    destindex_copy_kv(K, Dest_loc, Out)

    # Verify the result (for simplicity, just print the first element of each tensor)
    print("K[0, 0, 0, :10]:", K[0, 0, 0, :10])
    print("Dest_loc[0, 0, 0]:", Dest_loc[0, 0, 0])
    print("Out[0, 0, 0, :10]:", Out[0, 0, 0, :10])
