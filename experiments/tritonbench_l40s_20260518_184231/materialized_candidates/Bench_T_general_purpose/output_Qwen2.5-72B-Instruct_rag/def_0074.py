import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def normalize_kernel(
    x,
    norm,
    x_row_stride,
    n_cols,
    p,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    x_ptr = x + prog_id * x_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)

    norm_p = tl.sum(tl.abs(xf) ** p, 0) ** (1.0 / p)
    norm_val = tl.max(norm_p, eps)
    norm[prog_id] = norm_val

    normalized_x = (xf / norm_val).to(x.dtype)
    tl.store(x_ptr + offsets, normalized_x, mask=offsets < n_cols)

@triton.jit
def cosine_similarity_kernel(
    x1,
    x2,
    similarity,
    x1_row_stride,
    x2_row_stride,
    n_cols,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    x1_ptr = x1 + prog_id * x1_row_stride
    x2_ptr = x2 + prog_id * x2_row_stride

    x1 = tl.load(x1_ptr + offsets, mask=offsets < n_cols)
    x2 = tl.load(x2_ptr + offsets, mask=offsets < n_cols)
    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)

    dot_product = tl.sum(x1f * x2f, 0)
    norm_x1 = tl.max(tl.sqrt(tl.sum(x1f * x1f, 0)), eps)
    norm_x2 = tl.max(tl.sqrt(tl.sum(x2f * x2f, 0)), eps)

    similarity_val = dot_product / (norm_x1 * norm_x2)
    similarity[prog_id] = similarity_val

@torch.inference_mode()
def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    """
    Computes the cosine similarity between two normalized input tensors `x1` and `x2`.

    Args:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.
        dim (int, optional): The dimension along which to compute the cosine similarity. Default is 1.
        eps_similarity (float, optional): A small value to avoid division by zero in the similarity computation. Default is 1e-8.
        p_norm (float, optional): The p-norm value for normalization. Default is 2.
        eps_norm (float, optional): A small value to avoid division by zero in the normalization. Default is 1e-12.

    Returns:
        Tensor: The cosine similarity between the normalized tensors.
    """

    def _kernel_meta():
        device = x1.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    # Ensure x2 is broadcasted to match x1's shape
    x2 = x2.expand_as(x1)

    feat_size = x1.size(dim)
    seq_len = x1.numel() // x1.size(-1)
    x1_stride = x1.stride(dim)
    x2_stride = x2.stride(dim)

    BLOCK_N = triton.next_power_of_2(feat_size)
    norm_x1 = torch.empty((seq_len,), dtype=torch.float32, device=x1.device)
    norm_x2 = torch.empty((seq_len,), dtype=torch.float32, device=x2.device)

    # Normalize x1
    kernel_meta = _kernel_meta()
    grid = (seq_len,)
    normalize_kernel[grid](
        x1,
        norm_x1,
        x1_stride,
        feat_size,
        p_norm,
        eps_norm,
        feat_size,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    # Normalize x2
    normalize_kernel[grid](
        x2,
        norm_x2,
        x2_stride,
        feat_size,
        p_norm,
        eps_norm,
        feat_size,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    # Compute cosine similarity
    similarity = torch.empty((seq_len,), dtype=torch.float32, device=x1.device)
    cosine_similarity_kernel[grid](
        x1,
        x2,
        similarity,
        x1_stride,
        x2_stride,
        feat_size,
        eps_similarity,
        feat_size,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    return similarity

# Example usage
x1 = torch.randn((32, 64), device='cuda')
x2 = torch.randn((32, 64), device='cuda')
similarity = normalized_cosine_similarity(x1, x2)
print(similarity)
