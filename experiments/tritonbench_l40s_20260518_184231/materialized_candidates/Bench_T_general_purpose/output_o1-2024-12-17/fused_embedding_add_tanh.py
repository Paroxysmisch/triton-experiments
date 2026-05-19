import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _add_tanh_kernel(
    E_ptr, other_ptr, out_ptr,
    N,  # total number of elements in E/out
    OTHER_IS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_start = pid * BLOCK_SIZE
    offsets = offset_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load embedding values
    E_val = tl.load(E_ptr + offsets, mask=mask, other=0.0)

    # Load or broadcast 'other'
    if OTHER_IS_SCALAR:
        other_val = tl.load(other_ptr, mask=[True], other=0.0)
    else:
        other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)

    # Add and tanh
    S = E_val + other_val
    out_val = tl.math.tanh(S)

    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

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
    """
    fused_embedding_add_tanh(input_indices, weight, other, *,
                             padding_idx=None, max_norm=None, norm_type=2.0,
                             scale_grad_by_freq=False, sparse=False, out=None) -> Tensor

    Functional Description:
        Retrieves embeddings from 'weight' using 'input_indices', adds
        'other' (which must be broadcastable to the embedding shape),
        and applies tanh activation.

    Wrapper Entry Information:
        input_indices (LongTensor): Indices to lookup in 'weight'.
        weight (Tensor): Embedding matrix of shape (V, D).
        other (Tensor): Tensor broadcastable to matched embedding shape.
        padding_idx (int, optional): If specified, zero out gradient for that index.
        max_norm (float, optional): Renormalize embeddings if norm exceeds this.
        norm_type (float, optional): p-norm for max_norm. Default: 2.0
        scale_grad_by_freq (bool, optional): Scale gradient by freq. Default: False
        sparse (bool, optional): Sparse gradient w.r.t. weight. Default: False
        out (Tensor, optional): Output buffer. Ignored if None.

    Returns:
        A Tensor of the same shape as embeddings, containing tanh(E + other).
    """
    # 1. Retrieve embedding using PyTorch API (handles padding_idx, max_norm, etc.)
    E = F.embedding(
        input_indices, weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse
    )

    # 2. Check shape of the result and broadcast 'other'
    #    For simplicity, we rely on PyTorch to broadcast shapes beforehand,
    #    so we only fuse the add + tanh in Triton.
    broadcast_shape = E.shape
    other_broadcasted = torch.broadcast_to(other, broadcast_shape)

    # 3. Allocate or reuse the output tensor
    if out is None:
        out = torch.empty_like(E)

    # 4. Launch Triton kernel for (E + other) -> tanh
    N = E.numel()
    BLOCK_SIZE = 1024
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Decide if 'other' is scalar
    OTHER_IS_SCALAR = (other_broadcasted.numel() == 1)

    _add_tanh_kernel[grid](
        E, other_broadcasted, out,
        N,
        OTHER_IS_SCALAR=OTHER_IS_SCALAR,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
