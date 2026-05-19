import triton
import triton.language as tl

# Triton kernel function
@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K: tl.tensor,                # input tensor K of shape (batch_size, head_num, seq_len, depth)
    Dest_loc: tl.tensor,         # destination indices tensor of shape (seq_len,)
    Out: tl.tensor,              # output tensor Out of shape (batch_size, head_num, seq_len, depth)
    batch_size: tl.constexpr,      # batch size
    head_num: tl.constexpr,        # number of heads
    seq_len: tl.constexpr,         # sequence length
    depth: tl.constexpr,           # depth of the tensor
    BLOCK_HEAD: tl.constexpr       # block size for head dimension
):
    pid = tl.program_id(0)         # current index
    idx = pid % seq_len            # index within sequence length
    h = pid // seq_len             # head index

    # Compute pointers for source and destination
    k_ptrs = tl.make_range(batch_size * head_num * seq_len * depth) + \
             idx * batch_size * head_num * depth + \
             h * batch_size * depth + \
             tl.arange(batch_size) * head_num * depth

    o_ptrs = tl.make_range(batch_size * head_num * seq_len * depth) + \
             idx * batch_size * head_num * depth + \
             h * batch_size * depth + \
             tl.arange(batch_size) * head_num * depth

    # Load data from K and store it in Out
    for d in range(depth):
        k_val = tl.load(K + k_ptrs + d)
        tl.store(Out + o_ptrs + d, k_val)

# Wrapper function
def destindex_copy_kv(K, Dest_loc, Out, batch_size, head_num, seq_len, depth):
    # Validate tensor shapes
    assert K.shape == (batch_size, head_num, seq_len, depth)
    assert Out.shape == (batch_size, head_num, seq_len, depth)
    assert Dest_loc.shape == (seq_len,)

    # Compute BLOCK_HEAD as the next power of 2 of head_num
    BLOCK_HEAD = 1
    while BLOCK_HEAD < head_num:
        BLOCK_HEAD *= 2

    # Configure grid and block
    grid = (seq_len,)
    block = (BLOCK_HEAD,)

    # Launch kernel
    _fwd_kernel_destindex_copy_kv[grid, block](
        K, Dest_loc, Out, batch_size, head_num, seq_len, depth, BLOCK_HEAD
    )

# Example usage
if __name__ == "__main__":
    batch_size = 32
    head_num = 8
    seq_len = 128
    depth = 64

    K = tl.zeros((batch_size, head_num, seq_len, depth), dtype=tl.float32)
    Dest_loc = tl.arange(seq_len)
    Out = tl.zeros((batch_size, head_num, seq_len, depth), dtype=tl.float32)

    destindex_copy_kv(K, Dest_loc, Out, batch_size, head_num, seq_len, depth)
    print("Operation completed successfully.")
