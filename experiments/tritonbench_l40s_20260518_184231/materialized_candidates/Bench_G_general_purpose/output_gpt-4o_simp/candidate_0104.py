import triton
import triton.language as tl

# Triton kernel for the forward pass of attention mechanism
@triton.jit
def _attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, q_scale, k_scale, N_CTX, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute offsets for each block
    q_offset = pid * BLOCK_SIZE
    k_offset = 0  # Keys are shared across all queries
    v_offset = 0  # Values are shared across all queries

    # Load blocks of q, k, v
    q = tl.load(q_ptr + q_offset)
    k = tl.load(k_ptr + k_offset)
    v = tl.load(v_ptr + v_offset)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')

    # Compute attention for each block
    for n in range(0, N_CTX, BLOCK_SIZE):
        # Compute dot product and scale
        qk = tl.dot(q, k) * q_scale * k_scale
        
        # Update normalization factor
        l_i = tl.maximum(l_i, qk)

        # Compute exp and accumulate results
        exp_qk = tl.exp(qk - l_i[:, None])
        acc += tl.dot(exp_qk, v)

        # Update k and v offsets
        k_offset += BLOCK_SIZE
        v_offset += BLOCK_SIZE

        # Load next blocks of k, v
        k = tl.load(k_ptr + k_offset)
        v = tl.load(v_ptr + v_offset)

    # Normalize and write output
    acc = acc / tl.exp(l_i[:, None])
    tl.store(o_ptr + q_offset, acc)

# Wrapper function to call the Triton kernel
def forward(q, k, v, q_scale, k_scale, BLOCK_SIZE=128):
    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output tensor
    o = torch.empty_like(q)

    # Launch the Triton kernel
    grid = (q.shape[0] // BLOCK_SIZE,)
    _attn_fwd[grid](q, k, v, o, q_scale, k_scale, q.shape[1], BLOCK_SIZE=BLOCK_SIZE)

    return o
