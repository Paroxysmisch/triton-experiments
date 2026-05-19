import torch
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr, N, M, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Computes the cosine similarity between x1 and x2 along dim=1.

    Parameters:
    -----------
    x1_ptr : tl.tensor
        Pointer to the first input tensor in global memory.
    x2_ptr : tl.tensor
        Pointer to the second input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    N : int
        Number of elements in the batch dimension.
    M : int
        Number of elements in the feature dimension.
    eps : float
        Small value to prevent division by zero.
    BLOCK_SIZE : tl.constexpr
        Size of the block used in the Triton kernel grid, determines the number of elements processed per block.
    """
    batch_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets

    mask = idx < M
    x1_val = tl.load(x1_ptr + batch_id * M + idx, mask=mask, other=0.0)
    x2_val = tl.load(x2_ptr + batch_id * M + idx, mask=mask, other=0.0)

    dot_product = tl.sum(x1_val * x2_val, axis=0)
    norm_x1 = tl.sqrt(tl.sum(x1_val * x1_val, axis=0) + eps)
    norm_x2 = tl.sqrt(tl.sum(x2_val * x2_val, axis=0) + eps)

    similarity = dot_product / (norm_x1 * norm_x2)
    tl.store(output_ptr + batch_id * M + idx, similarity, mask=mask)

@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr, N, C, H, W, kernel_size, stride, padding, BLOCK_SIZE: tl.constexpr
):
    """
    Applies 2D average pooling to the input tensor.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    N : int
        Number of elements in the batch dimension.
    C : int
        Number of elements in the channel dimension.
    H : int
        Height of the input tensor.
    W : int
        Width of the input tensor.
    kernel_size : int
        Size of the pooling kernel.
    stride : int
        Stride of the pooling operation.
    padding : int
        Padding applied to the input tensor.
    BLOCK_SIZE : tl.constexpr
        Size of the block used in the Triton kernel grid, determines the number of elements processed per block.
    """
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_start_h = tl.program_id(2) * BLOCK_SIZE
    block_start_w = tl.program_id(3) * BLOCK_SIZE
    offsets_h = tl.arange(0, BLOCK_SIZE)
    offsets_w = tl.arange(0, BLOCK_SIZE)
    idx_h = block_start_h + offsets_h
    idx_w = block_start_w + offsets_w

    mask_h = (idx_h < H) & (idx_h >= 0)
    mask_w = (idx_w < W) & (idx_w >= 0)
    mask = mask_h & mask_w

    input_val = tl.load(input_ptr + batch_id * C * H * W + channel_id * H * W + idx_h * W + idx_w, mask=mask, other=0.0)
    sum_val = tl.sum(input_val, axis=0)
    avg_val = sum_val / (kernel_size * kernel_size)

    output_idx_h = (idx_h - padding) // stride
    output_idx_w = (idx_w - padding) // stride
    output_mask_h = (output_idx_h >= 0) & (output_idx_h < (H + 2 * padding - kernel_size) // stride + 1)
    output_mask_w = (output_idx_w >= 0) & (output_idx_w < (W + 2 * padding - kernel_size) // stride + 1)
    output_mask = output_mask_h & output_mask_w & mask

    tl.store(output_ptr + batch_id * C * ((H + 2 * padding - kernel_size) // stride + 1) * ((W + 2 * padding - kernel_size) // stride + 1) + channel_id * ((H + 2 * padding - kernel_size) // stride + 1) * ((W + 2 * padding - kernel_size) // stride + 1) + output_idx_h * ((W + 2 * padding - kernel_size) // stride + 1) + output_idx_w, avg_val, mask=output_mask)

### Wrapper Function

The wrapper function will manage the memory allocation, kernel launches, and tensor operations.
