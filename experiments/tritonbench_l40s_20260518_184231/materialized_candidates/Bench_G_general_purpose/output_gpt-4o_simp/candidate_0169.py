import triton
import triton.language as tl

# Triton kernel
@triton.jit
def _copy_to_kcache_seqlen_n_kernel(K, KCache, seq_lens, block_table, n_tokens, BLOCK_SIZE: tl.constexpr):
    # Define the program id and size
    pid = tl.program_id(axis=0)
    
    # Compute the start index in the cache for this program
    start_idx = seq_lens[pid] * n_tokens

    # Iterate over the block size
    for i in range(0, BLOCK_SIZE):
        # Compute the position in K
        k_pos = pid * BLOCK_SIZE + i

        # Compute the position in KCache
        cache_pos = start_idx + block_table[pid] * BLOCK_SIZE + i

        # Load the token from K
        token = tl.load(K + k_pos)

        # Store the token in KCache
        tl.store(KCache + cache_pos, token)

# Python wrapper
def copy_k_to_blocked_cache(K, KCache, seq_lens, block_table, n_tokens, block_size):
    # Ensure input dimensions are compatible
    assert K.shape[0] == len(seq_lens), "Incompatible input dimensions"
    assert len(block_table) == len(seq_lens), "Block table size must match sequence length size"

    # Determine the number of execution warps
    num_warps = min(4, (K.shape[0] + block_size - 1) // block_size)

    # Launch the Triton kernel
    grid = (K.shape[0],)
    _copy_to_kcache_seqlen_n_kernel[grid](K, KCache, seq_lens, block_table, n_tokens, BLOCK_SIZE=block_size, num_warps=num_warps)

# Example usage
import torch

# Example tensor sizes and parameters
K = torch.randn(1024, dtype=torch.float32).cuda()
KCache = torch.zeros_like(K).cuda()
seq_lens = torch.tensor([0, 1, 2, 3], dtype=torch.int32).cuda()
block_table = torch.tensor([0, 1, 2, 3], dtype=torch.int32).cuda()
n_tokens = 4
block_size = 256

# Call the function
copy_k_to_blocked_cache(K, KCache, seq_lens, block_table, n_tokens, block_size)
