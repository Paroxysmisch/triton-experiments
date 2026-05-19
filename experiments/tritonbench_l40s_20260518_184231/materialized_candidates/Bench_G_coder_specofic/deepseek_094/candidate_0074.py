c
__device__ void _copy_to_kvcache_seqlen1_kernel(
    uint32_t* K,
    uint32_t* V,
    uint32_t* KCache,
    uint32_t* VCache,
    uint32_t* block_tables,
    uint32_t num_blocks,
    uint32_t num_kv_heads,
    uint32_t block_size,
    uint32_t head_dim,
    uint32_t stride_length,
    uint32_t sequence_index,
    uint32_t head_index
) {
    uint32_t block_id = block_tables[sequence_index * num_kv_heads + head_index];
    uint32_t offset = block_id * num_kv_heads * block_size * head_dim;
    uint32_t kv_offset = sequence_index * num_kv_heads * block_size * head_dim;

    for (uint32_t i = threadIdx.x; i < block_size * head_dim; i += blockDim.x) {
        KCache[offset + i] = K[kv_offset + i];
        VCache[offset + i] = V[kv_offset + i];
    }
}
