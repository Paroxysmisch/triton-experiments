# Assuming you have the necessary tensors and parameters
batch_size = 32
head_dim = 12
kcache_x = 64
SEQLEN_MAX = 128
format = 1  # Using the new cache format

# Example tensors
k = ...  # Input key tensor
cache = ...  # Cache tensor
seqlen = ...  # Sequence length tensor

# Call the wrapper function
copy_k_to_blocked_cache(k, cache, seqlen, batch_size, head_dim, kcache_x, format, SEQLEN_MAX)
