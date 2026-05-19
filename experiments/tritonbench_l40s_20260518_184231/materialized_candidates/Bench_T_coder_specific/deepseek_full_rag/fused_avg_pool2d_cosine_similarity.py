import triton
import torch
import triton.language as tl
from torch.nn.functional import cosine_similarity
from torch.nn.functional import avg_pool2d

@triton.jit
def cosine_similarity_kernel(x1, x2, dim, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    # Compute cosine similarity
    mid = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    x1_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)
    x2_norm = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i in range(0, BLOCK_M):
        for j in range(0, BLOCK_M):
            mid[i, j] += tl.sum(x1[i * BLOCK_D : i * BLOCK_D + BLOCK_D] * x2[j * BLOCK_D : j * BLOCK_D + BLOCK_D])
        x1_norm[i] += tl.sum(x1[i * BLOCK_D : i * BLOCK_D + BLOCK_D] * x1[i * BLOCK_D : i * BLOCK_D + BLOCK_D])
        x2_norm[i] += tl.sum(x2[i * BLOCK_D : i * BLOCK_D + BLOCK_D] * x2[i * BLOCK_D : i * BLOCK_D + BLOCK_D])

    cos_sim = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    for i in range(0, BLOCK_M):
        for j in range(0, BLOCK_M):
            cos_sim[i, j] = mid[i, j] / ((x1_norm[i] * x2_norm[j]) + eps) ** 0.5

    return cos_sim


@triton.jit
def unsqueeze_kernel(x, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    # Add a singleton dimension
    return x[0 * BLOCK_D : 1 * BLOCK_D]


@triton.jit
def avg_pool2d_kernel(x, kernel_size, stride, padding, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    # Apply 2D average pooling
    return tl.sum(x, axis=0) / (kernel_size * kernel_size)


def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    if stride is None:
        stride = kernel_size

    assert x1.shape[0] == x2.shape[0], "Batch dimensions must be equal"
    assert x1.is_contiguous(), "Tensor x1 must be contiguous"
    assert x2.is_contiguous(), "Tensor x2 must be contiguous"

    batch_size, dim = x1.shape
    x1 = x1.view(batch_size, -1)
    x2 = x2.view(batch_size, -1)

    cosine_similarity_out = torch.empty((batch_size, batch_size), device=x1.device, dtype=torch.float32)

    num_blocks = triton.cdiv(dim, BLOCK_D)
    grid = (num_blocks, num_blocks)

    with torch.cuda.device(x1.device.index):
        cosine_similarity_kernel[grid](x1, x2, dim, eps, BLOCK_M=num_blocks, BLOCK_D=BLOCK_D)
        x1_unsqueeze = unsqueeze_kernel(x1, BLOCK_M=num_blocks, BLOCK_D=BLOCK_D)
        x2_unsqueeze = unsqueeze_kernel(x2, BLOCK_M=num_blocks, BLOCK_D=BLOCK_D)
        avg_pool2d_kernel[(batch_size, )](x1_unsqueeze, kernel_size, stride, padding, BLOCK_N=batch_size, BLOCK_D=BLOCK_D)
        avg_pool2d_kernel[(batch_size, )](x2_unsqueeze, kernel_size, stride, padding, BLOCK_N=batch_size, BLOCK_D=BLOCK_D)

    return cosine_similarity_out
