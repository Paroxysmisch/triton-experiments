import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    n_heads, head_dim, n_ctx,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    # Compute the batch and head indices
    bid = pid // n_heads
    hid = pid % n_heads

    # Pointers to the query, key, and value matrices
    Q_ptr = Q + bid * stride_qb + hid * stride_qh
    K_ptr = K + bid * stride_kb + hid * stride_kh
    V_ptr = V + bid * stride_vb + hid * stride_vh
    Out_ptr = Out + bid * stride_ob + hid * stride_oh

    # Allocate shared memory for the attention scores
    scores = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the context length in blocks
    for i in range(0, n_ctx, BLOCK_SIZE):
        # Load the query block
        q = tl.load(Q_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qd + tl.arange(0, head_dim))
        # Load the key block
        k = tl.load(K_ptr + i + tl.arange(0, BLOCK_SIZE) * stride_kd + tl.arange(0, head_dim)[:, None])

        # Compute the attention scores
        scores = tl.dot(q, k, allow_tf32=False)

        # Apply the softmax scaling
        max_score = tl.max(scores, 1)
        scores = scores - max_score[:, None]
        scores = tl.exp(scores)

        # Normalize the scores
        sum_scores = tl.sum(scores, 1)
        scores = scores / sum_scores[:, None]

        # Load the value block
        v = tl.load(V_ptr + i + tl.arange(0, BLOCK_SIZE) * stride_vd + tl.arange(0, head_dim)[:, None])

        # Compute the output
        out = tl.dot(scores, v, allow_tf32=False)

        # Store the output
        tl.store(Out_ptr + tl.arange(0, BLOCK_SIZE) * stride_od + tl.arange(0, head_dim)[:, None], out)

        # Move to the next block
        Q_ptr += BLOCK_SIZE * stride_qd
        K_ptr += BLOCK_SIZE * stride_kd
        V_ptr += BLOCK_SIZE * stride_vd
        Out_ptr += BLOCK_SIZE * stride_od

import torch

def context_attention_fwd_ppl_int8kv(Q, K, V, Out, n_heads, head_dim, n_ctx, BLOCK_SIZE=128, BLOCK_SIZE_HEAD=64):
    # Get the grid size
    grid = (Q.shape[0] * n_heads,)

    # Launch the kernel
    _fwd_kernel_int8kv[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        n_heads, head_dim, n_ctx,
        BLOCK_SIZE, BLOCK_SIZE_HEAD
    )

# Example usage
Q = torch.randint(-128, 128, (1, 8, 64), dtype=torch.int8, device='cuda')
K = torch.randint(-128, 128, (1, 8, 64), dtype=torch.int8, device='cuda')
V = torch.randint(-128, 128, (1, 8, 64), dtype=torch.int8, device='cuda')
Out = torch.zeros((1, 8, 64), dtype=torch.int8, device='cuda')

context_attention_fwd_ppl_int8kv(Q, K, V, Out, n_heads=8, head_dim=64, n_ctx=64)
