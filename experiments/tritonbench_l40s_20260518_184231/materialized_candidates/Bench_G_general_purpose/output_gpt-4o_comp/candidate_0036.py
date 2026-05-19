import triton
import triton.language as tl
import torch

@triton.jit
def _score_kernel(Q, K, M, Out, stride_qm, stride_kn, stride_out, sm_scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Define block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define block start positions
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Create block indices
    offsets_m = start_m + tl.arange(0, BLOCK_M)
    offsets_n = start_n + tl.arange(0, BLOCK_N)

    # Load Q and K blocks
    q = tl.load(Q + offsets_m[:, None] * stride_qm, mask=offsets_m[:, None] < Q.shape[0], other=0.0)
    k = tl.load(K + offsets_n[None, :] * stride_kn, mask=offsets_n[None, :] < K.shape[1], other=0.0)

    # Compute QK^T
    qk = tl.dot(q, k)

    # Scale by sm_scale
    qk_scaled = qk * sm_scale

    # Load mask and apply
    mask = tl.load(M + offsets_m[:, None] * M.shape[1] + offsets_n[None, :], mask=offsets_m[:, None] < M.shape[0], other=0.0)
    qk_scaled = tl.where(mask, qk_scaled, float('-inf'))

    # Store result in Out
    tl.store(Out + offsets_m[:, None] * stride_out + offsets_n[None, :], qk_scaled)

def get_score(Q, K, M, BLOCK_M=64, BLOCK_N=64):
    # Check device
    assert Q.is_cuda and K.is_cuda and M.is_cuda, "All inputs must be on GPU."

    # Calculate dimensions
    num_heads, seq_len_q, d_k = Q.shape
    _, seq_len_k, _ = K.shape

    # Prepare output tensor
    Out = torch.empty((num_heads, seq_len_q, seq_len_k), device=Q.device, dtype=Q.dtype)

    # Define grid size
    grid = (triton.cdiv(seq_len_q, BLOCK_M), triton.cdiv(seq_len_k, BLOCK_N))

    # Calculate sm_scale
    sm_scale = 1.0 / (d_k ** 0.5)

    # Execute kernel
    try:
        _score_kernel[grid](
            Q, K, M, Out,
            Q.stride(1), K.stride(1), Out.stride(1),
            sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
    except triton.OutOfResources as e:
        # Reduce block size and retry
        BLOCK_M //= 2
        BLOCK_N //= 2
        if BLOCK_M < 1 or BLOCK_N < 1:
            raise RuntimeError("Block size too small to execute kernel.") from e
        return get_score(Q, K, M, BLOCK_M, BLOCK_N)

    return Out

# Example usage
# Q, K, M should be prepared as torch tensors on GPU
# Q = torch.randn((num_heads, seq_len_q, d_k), device='cuda')
# K = torch.randn((num_heads, seq_len_k, d_k), device='cuda')
# M = torch.ones((num_heads, seq_len_q, seq_len_k), device='cuda', dtype=torch.bool)
# Out = get_score(Q, K, M)
