import triton
import triton.language as tl
import torch

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr, HAS_SEQ_IDX: tl.constexpr,
    seq_idx_ptr=None, seq_idx_stride=None
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (M * N)
    chunk_id = (pid // N) % M
    group_id = pid % N

    # Define the starting point for each block
    offs_am = batch_id * stride_am + chunk_id * BLOCK_SIZE_M
    offs_bn = group_id * BLOCK_SIZE_N
    offs_ak = batch_id * stride_ak + chunk_id * BLOCK_SIZE_K

    # Create pointers for sub-matrices
    a_ptrs = a_ptr + offs_am + offs_ak[:, None]
    b_ptrs = b_ptr + offs_ak[:, None] + offs_bn

    # Initialize accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs + k * stride_ak)
        b = tl.load(b_ptrs + k * stride_bk)
        acc += tl.dot(a, b)

    # Apply causal mask if needed
    if IS_CAUSAL:
        mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= tl.arange(0, BLOCK_SIZE_N)
        acc = tl.where(mask, acc, 0.0)

    # Apply sequence index mask if needed
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + batch_id * seq_idx_stride)
        mask = tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_idx[:, None]
        acc = tl.where(mask, acc, 0.0)

    # Store the result
    out_ptrs = out_ptr + offs_am + offs_bn
    tl.store(out_ptrs, acc)

def _bmm_chunk_fwd(
    a, b, out,
    block_size_m, block_size_n, block_size_k,
    is_causal=False, has_seq_idx=False, seq_idx=None
):
    # Ensure input tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    out = out.contiguous()

    # Get shapes and strides
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]
    stride_am, stride_ak = a.stride()[-2:]
    stride_bk, stride_bn = b.stride()[-2:]
    stride_cm, stride_cn = out.stride()[-2:]

    # Define grid size
    grid = (M * N,)

    # Launch the kernel
    triton._bmm_chunk_fwd_kernel[grid](
        a, b, out,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        block_size_m, block_size_n, block_size_k,
        is_causal, has_seq_idx,
        seq_idx if has_seq_idx else None, seq_idx.stride(0) if has_seq_idx else None
    )

# Example usage
a = torch.randn(64, 128, 256, device='cuda')
b = torch.randn(64, 256, 128, device='cuda')
out = torch.zeros(64, 128, 128, device='cuda')
_bmm_chunk_fwd(a, b, out, 16, 16, 16, is_causal=True)
