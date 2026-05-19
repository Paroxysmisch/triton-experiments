import torch
import triton
import triton.language as tl

# Define the Triton kernel
_fwd_kernel_destindex_copy_kv = triton.compile(
    _fwd_kernel_destindex_copy_kv,
    {
        "seq_len": triton.Infer(),
        "batch_size": triton.Infer(),
        "head_num": triton.Infer(),
        "head_dim": triton.Infer(),
        "BLOCK_HEAD": triton.Infer(),
        "BLOCK_DMODEL": triton.Infer(),
    },
    num_warps=4,
    num_stages=2,
)

def destindex_copy_kv(K, DestLoc, Out):
    # Ensure the input tensors have the correct shapes
    assert K.shape == Out.shape, "Input and output tensors must have the same shape"
    assert DestLoc.shape == (K.shape[0],), "Destination indices tensor must have the same length as the sequence length"

    # Get the tensor dimensions
    seq_len, batch_size, head_num, head_dim = K.shape

    # Determine the block sizes
    BLOCK_HEAD = 2 ** tl.next_power_of_two(head_num)
    BLOCK_DMODEL = 2 ** tl.next_power_of_two(head_dim)

    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_kv(
        K,
        DestLoc,
        Out,
        seq_len=seq_len,
        batch_size=batch_size,
        head_num=head_num,
        head_dim=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        grid=(seq_len,),
        block=(BLOCK_SIZE,),
    )

# Example usage
if __name__ == "__main__":
    # Create example tensors
    seq_len = 10
    batch_size = 2
    head_num = 4
    head_dim = 8

    K = torch.randn(seq_len, batch_size, head_num, head_dim, device="cuda")
    DestLoc = torch.randint(0, batch_size, (seq_len,), device="cuda")
    Out = torch.zeros_like(K, device="cuda")

    # Call the wrapper function
    destindex_copy_kv(K, DestLoc, Out)

    # Print the result
    print(Out)
