import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K_ptr,  # Pointer to the input tensor K
    Out_ptr,  # Pointer to the output tensor Out
    DestLoc_ptr,  # Pointer to the destination indices tensor DestLoc
    batch_size,  # Batch size
    head_num,  # Number of heads
    head_dim,  # Head dimension
    seq_len,  # Sequence length
    BLOCK_HEAD: tl.constexpr,  # Block size for head dimension
    BLOCK_DMODEL: tl.constexpr  # Block size for head dimension
):
    # Compute the starting index for the current block
    pid = tl.program_id(axis=0)
    cur_index = pid

    # Compute the offsets for head and depth dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute the destination index from DestLoc
    dest_index = tl.load(DestLoc_ptr + cur_index)

    # Ensure the destination index is within bounds
    dest_index = tl.where(dest_index < batch_size, dest_index, 0)

    # Compute the base pointers for K and Out
    K_base_ptr = K_ptr + cur_index * head_num * head_dim
    Out_base_ptr = Out_ptr + dest_index * head_num * head_dim

    # Load the data from K
    K_block = tl.load(K_base_ptr + offs_h[:, None] * head_dim + offs_d[None, :], mask=offs_h[:, None] < head_num, other=0.0)

    # Store the data to Out
    tl.store(Out_base_ptr + offs_h[:, None] * head_dim + offs_d[None, :], K_block, mask=offs_h[:, None] < head_num)

import torch
import triton
import triton.language as tl

def destindex_copy_kv(K, Out, DestLoc):
    # Ensure the input tensors are on the same device
    assert K.device == Out.device == DestLoc.device, "All tensors must be on the same device"
    
    # Ensure the input tensors have the correct dimensions
    batch_size, head_num, head_dim = K.shape
    seq_len = DestLoc.shape[0]
    assert Out.shape == (batch_size, head_num, head_dim), "Output tensor must have the same shape as the input tensor K"

    # Define the block sizes
    BLOCK_HEAD = 32
    BLOCK_DMODEL = 32

    # Define the grid size
    grid = (seq_len,)

    # Launch the kernel
    _fwd_kernel_destindex_copy_kv[grid](
        K, Out, DestLoc,
        batch_size, head_num, head_dim, seq_len,
        BLOCK_HEAD, BLOCK_DMODEL
    )

# Example usage
if __name__ == "__main__":
    # Create example tensors
    batch_size = 4
    head_num = 8
    head_dim = 64
    seq_len = 10

    K = torch.randn((seq_len, head_num, head_dim), device="cuda")
    Out = torch.zeros((batch_size, head_num, head_dim), device="cuda")
    DestLoc = torch.randint(0, batch_size, (seq_len,), device="cuda")

    # Call the wrapper function
    destindex_copy_kv(K, Out, DestLoc)

    # Print the result
    print(Out)
