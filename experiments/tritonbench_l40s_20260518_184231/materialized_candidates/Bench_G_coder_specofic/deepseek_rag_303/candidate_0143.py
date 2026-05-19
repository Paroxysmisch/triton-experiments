import torch
import triton
import triton.language as tl

from .triton_util import calculate_lastdim_num_blocks, get_torch_dtype_from_triton, get_preferred_dtype
from typing import Optional, Tuple

@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    seq_idx_ptr,
    # strides
    stride_a_batch,
    stride_a_chunk,
    stride_a_group,
    stride_a_head,
    stride_a_m,
    stride_a_k,
    stride_b_batch,
    stride_b_chunk,
    stride_b_group,
    stride_b_head,
    stride_b_k,
    stride_b_n,
    stride_out_batch,
    stride_out_chunk,
    stride_out_group,
    stride_out_head,
    stride_out_m,
    stride_out_n,
    stride_seq_idx_batch,
    stride_seq_idx_chunk,
    stride_seq_idx_group,
    stride_seq_idx_seq,
    # other flags
    HAS_SEQ_IDX: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    # matrix shapes
    chunk_size_m,
    chunk_size_k,
    chunk_size_n,
    batch,
    nchunks,
    nheads,
    ngroups_per_chunk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr
):
    """
    Kernel for computing a batch matrix multiplication
    A_batch x A_m x A_k * B_batch x B_k x B_n = Out_batch x Out_m x Out_n
    with additition of chunking, sequqnce indexing, and causal masking
    """
    # get linear indices
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_h = tl.program_id(axis=3)
    pid_g = pid_b // nchunks
    pid_b = pid_b % nchunks
    num_pid_n = tl.cdiv(chunk_size_n, BLOCK_SIZE_N)
    pid_n = pid_h % num_pid_n
    pid_m = tl.cdiv(pid_h, num_pid_n)
    num_pid_m = tl.cdiv(chunk_size_m, BLOCK_SIZE_M)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(chunk_size_k, BLOCK_SIZE_K)
    # parallel over k
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    a_ptr += pid_b * stride_a_batch + pid_g * stride_a_group + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_g * stride_b_group + pid_h * stride_b_head
    for k in range(0, num_pid_k):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        # Fetch data (Note: this depends on the specific use case for now)
        a = tl.load(a_ptr + offs_k[:, None] * stride_a_k + offs_m[None, :] * stride_a_m)
        b = tl.load(b_ptr + offs_k[None, :] * stride_b_k + offs_n[:, None] * stride_b_n)
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            a = a * causal_mask
        acc += tl.dot(a, b)

    orig_m = pid_m * BLOCK_SIZE_M
    orig_n = pid_n * BLOCK_SIZE_N

    # store the result
    out_ptr += pid_b * stride_out_batch + pid_g * stride_out_group + pid_h * stride_out_head
    out_ptr += orig_m * stride_out_m + orig_n * stride_out_n
    if HAS_SEQ_IDX:
        # make sure we not write out of our "window" of actual output
        valid_mask_out = (offs_m[:, None] < chunk_size_m) & (offs_n[None, :] < chunk_size_n)
        tl.store(out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n, acc, mask=valid_mask_out)
    else:
        tl.store(out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n, acc)


def _bmm_chunk_fwd(
    a,
    b,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    chunk_size: Optional[int] = None,
    seq_idx: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    """
    Wrapper over Triton bmm_chunk_fwd_kernel

    Args:
        a (torch.Tensor): Input tensor
        b (torch.Tensor): Input tensor
        attn_mask (Optional[torch.Tensor]): Optional attention mask of shape (batch, seq_len, chunk_size, chunk_size)
        dropout_p (float):probability of an element to be zeroed
        is_causal (bool): Wheter the attention should be causal
        chunk_size (Optional[int]): Chunk size along seq_len dimension
        seq_idx (Optional[torch.Tensor]): Sequence index of shape (batch, chunk, seq)
        output (Optional[torch.Tensor): Output buffer

    Returns:
        torch.Tensor: output tensor
    """
    has_chunks = chunk_size is not None
    a_ndim = a.dim()
    b_ndim = b.dim()

    if a_ndim != 4 or b_ndim != 4:
        raise RuntimeError("Both input tensors must be 4D")

    if not (a_ndim == b_ndim == 4):
        raise RuntimeError("Both input tensors must be 4D")

    if not (a.size(2) == b.size(3) and a.size(3) == b.size(2)):
        raise RuntimeError("Incompatible dimensions")

    if dropout_p > 0.0 or (attn_mask is not None) or a.stride(3) > 1 or b.stride(2) > 1:
        raise RuntimeError("Non-trivial dropout, attention or strided input is not supported")

    batch, seqlen, nheads, head_dim = a.shape
    _, _, ngroups_per_chunk, _ = b.shape

    if has_chunks:
        if seqlen % chunk_size != 0:
