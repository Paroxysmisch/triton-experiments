import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple

@triton.jit
def embedding_kernel(
    weights_ptr, weights_batch_stride, weights_out_stride,
    out_ptr, out_batch_stride, out_seq_stride, out_embed_stride,
    n_rows, n_cols,
    BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr
):
    pid_b = tl.program_id(1)
    pid_m = tl.program_id(0)
    weights_ptr += pid_b * weights_batch_stride
    out_ptr += pid_b * out_batch_stride

    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows

    offs_n = tl.arange(0, BLOCK_NN)
    mask_n = offs_n < n_cols

    mask = mask_m[:, None] & mask_n[None, :]

    nnz = tl.sum(mask.to(tl.int32))
    vec_ptr = tl.zeros([BLOCK_NN], dtype=tl.int64)
    vec_ptr += tl.argmin(mask.to(tl.int32) * (n_cols - offs_n[None, :]) + (n_cols - offs_n[None, :]) * (n_cols - offs_n[None, :]) + 1, axis=1)

    token_ids = offs_m[:, None] * n_cols + vec_ptr[None, :]
    weights_ptrs = weights_ptr + token_ids * weights_out_stride
    embedding_vectors = tl.load(weights_ptrs, mask=mask, other=0.0)

    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_NN)
    out_ptrs = out_ptr + offs_m[:, None] * out_seq_stride + offs_n[None, :] * out_embed_stride
    tl.store(out_ptrs, embedding_vectors, mask=mask)

def embedding(
    weights: Tensor,
    token_ids: Tensor,
    max_seq_length: int,
    padding_idx: Optional[int] = None,
) -> Tensor:
    BLOCK_N = 32
    BLOCK_NN = 64

    if padding_idx is not None:
        assert padding_idx < weights.size(1), "padding_idx must be less than the number of columns in weights"

    if token_ids.ndim == 1:
        batch_dim = None
        n_rows = token_ids.size(0)
        token_ids = token_ids.unsqueeze(0)
    elif token_ids.ndim == 2:
        batch_dim = 0
        n_rows, n_cols = token_ids.shape
    else:
        raise ValueError("token_ids must be 1- or 2-dimensional")

    n_cols = weights.size(1)
    out = token_ids.new_empty((n_rows, max_seq_length, n_cols) if batch_dim is None else (n_rows, n_cols))
    n_tokens = n_rows * n_cols

    grid = lambda meta: (triton.cdiv(n_tokens, meta["BLOCK_N"] * meta["BLOCK_NN"]), 1)

    with torch.cuda.device(token_ids.device):
        embedding_kernel[grid](
            weights, weights.stride(0), weights.stride(1),
            out, out.stride(0), out.stride(1) if batch_dim is None else out.stride(2),
            out, out.stride(0), out.stride(1) if batch_dim is None else out.stride(2),
            n_rows, n_cols,
            BLOCK_N=BLOCK_N, BLOCK_NN=BLOCK_NN
        )

    if padding_idx is not None:
        out.masked_fill_(token_ids == padding_idx, 0.0)

    return out.squeeze(dim=0) if batch_dim is None else out
