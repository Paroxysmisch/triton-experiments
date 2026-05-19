import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr, N, eps: tl.constexpr
):
    """
    Compute cosine similarity between x1 and x2 along the second dimension.

    Parameters:
    -----------
    x1_ptr : tl.tensor
        Pointer to the first input tensor in global memory.
    x2_ptr : tl.tensor
        Pointer to the second input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    N : int
        Size of the dimension along which to compute cosine similarity.
    eps : float
        Small value to prevent division by zero.
    """
    idx = tl.program_id(0)
    offsets = tl.arange(0, N)
    x1 = tl.load(x1_ptr + idx * N + offsets)
    x2 = tl.load(x2_ptr + idx * N + offsets)

    dot_product = tl.sum(x1 * x2, axis=0)
    norm_x1 = tl.sqrt(tl.sum(x1 * x1, axis=0) + eps)
    norm_x2 = tl.sqrt(tl.sum(x2 * x2, axis=0) + eps)

    cosine_similarity = dot_product / (norm_x1 * norm_x2 + eps)
    tl.store(output_ptr + idx, cosine_similarity)

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the cosine similarity between x1 and x2 along dimension 1, adds a singleton dimension, and applies 2D average pooling.

    Parameters:
    -----------
    x1 : torch.Tensor
        The first input tensor.
    x2 : torch.Tensor
        The second input tensor.
    kernel_size : int
        Size of the pooling kernel.
    stride : int, optional
        Stride of the pooling operation. Defaults to the value of kernel_size.
    padding : int, optional
        Padding added to the input.
    eps : float, optional
        Small value to prevent division by zero in cosine similarity.

    Returns:
    --------
    torch.Tensor
        The resulting tensor after computing cosine similarity, adding a singleton dimension, and applying average pooling.
    """
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape."
    B, N = x1.shape

    # Prepare output tensor
    cosine_similarity_output = torch.empty((B,), device=x1.device, dtype=x1.dtype)

    # Launch Triton kernel
    grid = (B,)
    cosine_similarity_kernel[grid](
        x1_ptr=x1, x2_ptr=x2, output_ptr=cosine_similarity_output, N=N, eps=eps
    )

    # Add singleton dimension
    cosine_similarity_output = cosine_similarity_output.unsqueeze(1)

    # Apply 2D average pooling
    stride = kernel_size if stride is None else stride
    pooled_output = F.avg_pool2d(cosine_similarity_output, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding))

    return pooled_output
