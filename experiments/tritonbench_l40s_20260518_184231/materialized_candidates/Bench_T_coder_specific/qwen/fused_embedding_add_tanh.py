import torch
import triton

def fused_embedding_add_tanh(
    input_indices,
    weight,
    other,
    *,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    out=None
):
    # Validate inputs
    if input_indices.dtype != torch.long:
        raise ValueError("input_indices must be of type LongTensor")
    if weight.dim() != 2 or weight.size(1) != other.size(-1):
        raise ValueError("weight must be of shape (V, D) and other must be broadcastable to (N, D)")
    if padding_idx is not None and padding_idx >= weight.size(0):
        raise ValueError("padding_idx must be less than the vocabulary size")

    # Initialize output tensor if not provided
    if out is None:
        out = torch.empty_like(other)

    # Set up Triton kernel arguments
    block_size = 32
    grid_size = (len(input_indices) + block_size - 1) // block_size

    # Launch Triton kernel
    fused_embedding_add_tanh_kernel[grid_size, block_size](
        input_indices=input_indices.contiguous(),
        weight=weight.contiguous(),
        other=other.contiguous(),
        output=out.contiguous(),
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )

    return out
