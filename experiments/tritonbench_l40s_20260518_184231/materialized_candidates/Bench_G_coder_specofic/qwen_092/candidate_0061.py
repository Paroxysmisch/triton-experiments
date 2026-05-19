triton
@triton.jit
def _sgmv_expand_slice_kernel(
    A_ptr, B_ptr, C_ptr, lora_indices_ptr, weights_ptr,
    A_shape, B_shape, C_shape, lora_indices_shape, weights_shape,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(A_shape[0], BLOCK_M)
    
    row = pid * BLOCK_M
    col = pid % BLOCK_N
    
    A = tl.load(A_ptr + row * A_shape[1] + col, mask=col < A_shape[1], eviction_policy=tl.EVICT_FALSE)
    B = tl.load(B_ptr + row * B_shape[1] + col, mask=col < B_shape[1], eviction_policy=tl.EVICT_FALSE)
    C = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, A_shape[2], BLOCK_K):
        A_block = tl.load(A_ptr + row * A_shape[1] + k, mask=(k < A_shape[2]) & (col < A_shape[1]), eviction_policy=tl.EVICT_FALSE)
        B_block = tl.load(B_ptr + row * B_shape[1] + k, mask=(k < B_shape[2]) & (col < B_shape[1]), eviction_policy=tl.EVICT_FALSE)
        C += tl.dot(A_block, B_block)
    
    tl.store(C_ptr + row * C_shape[1] + col, C, mask=col < C_shape[1])
