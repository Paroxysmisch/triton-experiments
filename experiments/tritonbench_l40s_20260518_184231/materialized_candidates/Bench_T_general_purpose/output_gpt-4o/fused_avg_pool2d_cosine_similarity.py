import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(x1_ptr, x2_ptr, output_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load elements
    x1 = tl.load(x1_ptr + offsets, mask=offsets < n_elements, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute dot product
    dot_product = tl.sum(x1 * x2, axis=0)

    # Compute norms
    norm_x1 = tl.sqrt(tl.sum(x1 * x1, axis=0) + eps)
    norm_x2 = tl.sqrt(tl.sum(x2 * x2, axis=0) + eps)

    # Compute cosine similarity
    cosine_sim = dot_product / (norm_x1 * norm_x2 + eps)

    # Store result
    tl.store(output_ptr + pid, cosine_sim)

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size

    # Check that x1 and x2 have the same shape
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"

    # Calculate the number of elements in the dimension along which cosine similarity is computed
    n_elements = x1.size(1)

    # Prepare output tensor for cosine similarity
    cosine_sim = torch.empty(x1.size(0), device=x1.device, dtype=x1.dtype)

    # Launch Triton kernel to compute cosine similarity
    grid = (x1.size(0),)
    cosine_similarity_kernel[grid](x1, x2, cosine_sim, n_elements, eps, BLOCK_SIZE=1024)

    # Add a singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)

    # Apply 2D average pooling
    pooled_output = F.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)

    return pooled_output
