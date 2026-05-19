import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(x1, x2, output_size, p, eps, keepdim, MASK):
    # Triton kernel for pairwise distance calculation
    # Adaptive average pooling is not explicitly applied in the kernel
    batch_size = x1.shape[0]
    channels = x1.shape[1]
    dim = x1.shape[2]

    height = tl.full([batch_size, channels, 1], dim, dtype=tl.int32)
    width = tl.full([batch_size, channels, 1], 1, dtype=tl.int32)

    x1 = tl.reshape(x1, [batch_size, channels, dim, 1])
    x2 = tl.reshape(x2, [batch_size, channels, 1, dim])

    m = tl.dot(x1, x2)
    m = tl.where(MASK, m, 0)

    x1 = tl.reshape(x1, [batch_size, channels, dim])
    x2 = tl.reshape(x2, [batch_size, channels, dim])

    x1 = x1 * x1
    x2 = x2 * x2
    x1 = tl.sum(x1, axis=2, keepdim=True)
    x2 = tl.sum(x2, axis=2, keepdim=True)

    x1 = tl.sqrt(x1 + eps)
    x2 = tl.sqrt(x2 + eps)

    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)

    output = x1 + x2.transpose([0, 1, 2]) - 2 * m
    output = tl.abs(output) if p == 1 else output

    if p != 2:
        output = tl.pow(output, p)

    output = tl.sum(output, axis=2)

    if keepdim is False:
        height = height.item()
        width = width.item()
        output = tl.reshape(output, [batch_size, channels, height, width])

    return output

def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
) -> torch.Tensor:
    # Function to apply adaptive average pooling and compute pairwise distance
    assert x1.dim() == 4 and x2.dim() == 4, "Input tensors must be 4D"
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    assert x1.is_contiguous() and x2.is_contiguous(), "Input tensors must be contiguous"

    batch_size, channels, height, width = x1.shape

    x1 = triton.language.adaptive_avg_pool2d(x1, output_size)
    x2 = triton.language.adaptive_avg_pool2d(x2, output_size)

    mask = (
        torch.arange(height, device=x1.device)[None, :] < height
    )  # mask for broadcasting
    mask = mask.reshape(1, 1, height, 1)
    mask = mask.repeat(batch_size, channels, 1, width)

    output = pairwise_distance_kernel(x1, x2, output_size, p, eps, keepdim, mask)

    return output
