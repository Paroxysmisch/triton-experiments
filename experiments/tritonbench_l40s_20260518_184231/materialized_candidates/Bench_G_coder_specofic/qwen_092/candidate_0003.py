import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'WARPS_PER_BLOCK': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'WARPS_PER_BLOCK': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'WARPS_PER_BLOCK': 8}, num_stages=1, num_warps=8),
    ],
    key=['n_batch', 'n_head', 'n_seq', 'n_embd'],
)
@triton.jit
def context_attention_fwd(
    Q, K, V, Out,
    stride_q, stride_k, stride_v, stride_out,
    n_batch, n_head, n_seq, n_embd,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WARPS_PER_BLOCK: tl.constexpr,
):
    # Get the block ID
    pid = tl.program_id(axis=0)
    bid = pid // (n_head * WARPS_PER_BLOCK)
    h = pid % n_head
    # Get the block offsets
    b = bid // (n_seq * WARPS_PER_BLOCK)
    s = (bid % (n_seq * WARPS_PER_BLOCK)) // WARPS_PER_BLOCK
    o = bid % WARPS_PER_BLOCK
    # Get the thread ID within the block
    tid = tl.thread_id(axis=0)
    # Get the sequence and head offsets
    q_offset = (b * stride_q + s * stride_out + h * n_embd) // 16
    k_offset = (b * stride_k + h * n_embd) // 16
    v_offset = (b * stride_v + h * n_embd) // 16
    out_offset = (b * stride_out + s * stride_out + h * n_embd) // 16
    # Load the query, key, and value
    q = tl.load(Q + q_offset + tid * 16, mask=tid < n_embd, other=0.0)
    k = tl.load(K + k_offset + tid * 16, mask=tid < n_embd, other=0.0)
    v = tl.load(V + v_offset + tid * 16, mask=tid < n_embd, other=0.0)
    # Compute the dot product and scale
    dot = tl.dot(q, k, allow_tf32=True)
    dot = dot * scale
    # Compute the softmax
    max_val = tl.max(dot, axis=1, keepdim=True)
    exp_dot = tl.exp(dot - max_val)
    sum_exp_dot = tl.sum(exp_dot, axis=1, keepdim=True)
    weights = exp_dot / sum_exp_dot
    # Compute the output
    out = tl.dot(weights, v, allow_tf32=True)
    # Store the output
    tl.store(Out + out_offset + tid * 16, out, mask=tid < n_embd)

def context_attention_forward(Q, K, V, Out, n_batch, n_head, n_seq, n_embd, scale):
    # Compute grid and block sizes
    grid = (n_batch * n_seq * n_head * WARPS_PER_BLOCK + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    context_attention_fwd[grid, BLOCK_SIZE](Q, K, V, Out,
                                          Q.stride(0), K.stride(0), V.stride(0), Out.stride(0),
                                          n_batch, n_head, n_seq, n_embd,
                                          scale, BLOCK_SIZE, WARPS_PER_BLOCK)
