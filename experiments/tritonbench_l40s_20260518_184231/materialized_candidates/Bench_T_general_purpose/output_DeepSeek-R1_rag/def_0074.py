import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def normalized_cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    x1_row_stride,
    x2_row_stride,
    output_row_stride,
    n_cols,
    p_norm,
    eps_norm,
    eps_similarity,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < n_cols

    # Load rows from x1 and x2
    x1_row = tl.load(x1_ptr + row_idx * x1_row_stride + offsets, mask=mask, other=0.0)
    x2_row = tl.load(x2_ptr + row_idx * x2_row_stride + offsets, mask=mask, other=0.0)

    # Convert to float32 for precision
    x1_f32 = x1_row.to(tl.float32)
    x2_f32 = x2_row.to(tl.float32)

    # Compute L_p norm for x1
    abs_x1 = tl.abs(x1_f32)
    pow_x1 = tl.pow(abs_x1, p_norm)
    sum_p_x1 = tl.sum(pow_x1, axis=0)
    inv_p = 1.0 / p_norm
    norm_x1 = tl.pow(sum_p_x1, inv_p)
    norm_x1 = tl.maximum(norm_x1, eps_norm)
    x1_normalized = x1_f32 / norm_x1

    # Compute L_p norm for x2
    abs_x2 = tl.abs(x2_f32)
    pow_x2 = tl.pow(abs_x2, p_norm)
    sum_p_x2 = tl.sum(pow_x2, axis=0)
    norm_x2 = tl.pow(sum_p_x2, inv_p)
    norm_x2 = tl.maximum(norm_x2, eps_norm)
    x2_normalized = x2_f32 / norm_x2

    # Compute dot product of normalized vectors
    dot_product = tl.sum(x1_normalized * x2_normalized, axis=0)

    # Compute L2 norms of normalized vectors
    l2_x1 = tl.sqrt(tl.sum(x1_normalized * x1_normalized, axis=0))
    l2_x2 = tl.sqrt(tl.sum(x2_normalized * x2_normalized, axis=0))

    # Apply max with eps_similarity to L2 norms
    l2_x1 = tl.maximum(l2_x1, eps_similarity)
    l2_x2 = tl.maximum(l2_x2, eps_similarity)

    # Compute similarity
    similarity = dot_product / (l2_x1 * l2_x2)

    # Convert back to original dtype
    output_dtype = x1_row.dtype
    similarity = similarity.to(output_dtype)

    # Store result
    output_row = output_ptr + row_idx * output_row_stride
    tl.store(output_row, similarity)

@torch.inference_mode()
def normalized_cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps_similarity: float = 1e-8,
    p_norm: float = 2,
    eps_norm: float = 1e-12,
) -> Tensor:
    # Broadcast x2 to x1's shape
    x2 = x2.broadcast_to(x1.shape)
    # Move the specified dimension to the last dimension
    x1_transposed = x1.movedim(dim, -1)
    x2_transposed = x2.movedim(dim, -1)
    # Flatten all dimensions except the last into a single dimension
    num_rows = x1_transposed.numel() // x1_transposed.size(-1)
    feature_dim = x1_transposed.size(-1)
    x1_2d = x1_transposed.reshape(num_rows, feature_dim).contiguous()
    x2_2d = x2_transposed.reshape(num_rows, feature_dim).contiguous()
    # Allocate output tensor
    output = torch.empty(num_rows, dtype=x1.dtype, device=x1.device)
    # Determine block size
    BLOCK_N = triton.next_power_of_2(feature_dim)
    # Kernel launch parameters
    grid = (num_rows,)
    def _kernel_meta():
        device = x1.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)
    kernel_meta = _kernel_meta()
    # Launch kernel
    normalized_cosine_similarity_kernel[grid](
        x1_2d,
        x2_2d,
        output,
        x1_2d.stride(0),
        x2_2d.stride(0),
        output.stride(0),
        feature_dim,
        p_norm,
        eps_norm,
        eps_similarity,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )
    # Reshape output to original shape without the specified dimension
    original_output_shape = list(x1.shape)
    del original_output_shape[dim]
    return output.reshape(original_output_shape)
