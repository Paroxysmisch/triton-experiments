import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_tanh_kernel(
    input_indices_ptr, weight_ptr, other_ptr, out_ptr,
    num_embeddings, embedding_dim, num_indices,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Calculate the start index for this block
    start_idx = pid * BLOCK_SIZE

    # Load indices for this block
    indices = tl.load(input_indices_ptr + start_idx, mask=start_idx < num_indices)

    # Load embeddings and other tensor
    embeddings = tl.zeros([BLOCK_SIZE, embedding_dim], dtype=tl.float32)
    other = tl.load(other_ptr + start_idx * embedding_dim, mask=start_idx < num_indices)

    for i in range(BLOCK_SIZE):
        if start_idx + i < num_indices:
            idx = indices[i]
            embedding = tl.load(weight_ptr + idx * embedding_dim, mask=idx < num_embeddings)
            embeddings[i, :] = embedding

    # Element-wise addition
    result = embeddings + other

    # Apply tanh activation
    result = tl.math.tanh(result)

    # Store the result
    tl.store(out_ptr + start_idx * embedding_dim, result, mask=start_idx < num_indices)

import torch

def fused_embedding_add_tanh(input_indices, weight, other, *,
                             padding_idx=None, max_norm=None, norm_type=2.0,
                             scale_grad_by_freq=False, sparse=False, out=None):
    # Check input types and shapes
    assert input_indices.dtype == torch.long, "input_indices must be a LongTensor"
    assert weight.dim() == 2, "weight must be a 2D tensor"
    assert other.dim() >= 1, "other must be at least 1D tensor"

    num_embeddings, embedding_dim = weight.shape
    num_indices = input_indices.numel()

    # Prepare output tensor
    if out is None:
        out = torch.empty((num_indices, embedding_dim), device=weight.device, dtype=weight.dtype)

    # Handle padding_idx
    if padding_idx is not None:
        input_indices = input_indices.clone()
        input_indices[input_indices == padding_idx] = 0

    # Normalize embeddings if max_norm is specified
    if max_norm is not None:
        with torch.no_grad():
            norms = weight.norm(p=norm_type, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm)
            weight.mul_(desired / (norms + 1e-7))

    # Launch Triton kernel
    BLOCK_SIZE = 128  # You can adjust this based on your GPU
    grid = (num_indices + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_embedding_add_tanh_kernel[grid](
        input_indices, weight, other, out,
        num_embeddings, embedding_dim, num_indices,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
