import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_tanh_kernel(
    input_indices_ptr,  # *shape, int32
    weight_ptr,         # (V, D), float32
    other_ptr,          # *shape, float32
    output_ptr,         # *shape, float32
    padding_idx,        # int32
    max_norm,           # float32
    norm_type,          # float32
    scale_grad_by_freq, # int32 (0 or 1)
    sparse,             # int32 (0 or 1)
    V,                  # int32
    D,                  # int32
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < V * D

    # Load input indices
    input_indices = tl.load(input_indices_ptr + offsets, mask=mask, other=0)

    # Load weights
    weight_offsets = input_indices * D + tl.arange(0, D)
    embeddings = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)

    # Apply max norm if specified
    if max_norm > 0.0:
        norms = tl.norm(embeddings, ord=norm_type, axis=1)
        norms = tl.where(norms > max_norm, max_norm / norms, 1.0)
        embeddings = embeddings * norms[:, None]

    # Load other tensor
    other_offsets = block_start + tl.arange(0, D)
    other = tl.load(other_ptr + other_offsets, mask=mask, other=0.0)

    # Element-wise addition
    sum = embeddings + other

    # Apply tanh activation
    output = tl.tanh(sum)

    # Store the result
    tl.store(output_ptr + other_offsets, output, mask=mask)

import torch
import triton
import triton.language as tl

def fused_embedding_add_tanh(input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None):
    # Ensure input tensors are on the same device
    device = input_indices.device
    weight = weight.to(device)
    other = other.to(device)
    
    # Validate shapes
    assert input_indices.dtype == torch.int32, "input_indices must be of type int32"
    assert weight.dim() == 2, "weight must be a 2D tensor"
    assert other.shape == input_indices.shape, "other must have the same shape as input_indices"
    
    # Initialize output tensor
    if out is None:
        out = torch.empty_like(other, device=device)
    else:
        assert out.shape == other.shape, "out must have the same shape as other"
        assert out.device == device, "out must be on the same device as input_indices"
    
    # Convert optional parameters to appropriate types
    padding_idx = padding_idx if padding_idx is not None else -1
    max_norm = max_norm if max_norm is not None else -1.0
    scale_grad_by_freq = int(scale_grad_by_freq)
    sparse = int(sparse)
    
    # Launch the Triton kernel
    grid = (input_indices.numel() // 1024 + 1,)
    fused_embedding_add_tanh_kernel[grid](
        input_indices, weight, other, out,
        padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
        weight.shape[0], weight.shape[1], 1024
    )
    
    return out
