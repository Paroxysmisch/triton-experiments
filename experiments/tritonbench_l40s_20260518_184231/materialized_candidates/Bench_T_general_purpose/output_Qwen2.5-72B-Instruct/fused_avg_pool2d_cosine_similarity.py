import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr,
    n1, n2, d, stride1, stride2, stride3,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = batch_start + BLOCK_SIZE

    for i in range(batch_start, batch_end):
        if i < n1:
            sum_x1 = 0.0
            sum_x2 = 0.0
            dot_product = 0.0
            for j in range(d):
                x1_val = tl.load(x1_ptr + i * stride1 + j)
                x2_val = tl.load(x2_ptr + i * stride2 + j)
                dot_product += x1_val * x2_val
                sum_x1 += x1_val * x1_val
                sum_x2 += x2_val * x2_val
            norm_x1 = tl.sqrt(sum_x1 + eps)
            norm_x2 = tl.sqrt(sum_x2 + eps)
            similarity = dot_product / (norm_x1 * norm_x2)
            tl.store(output_ptr + i, similarity)

@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    n, c, h, w, out_h, out_w,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = batch_start + BLOCK_SIZE

    for i in range(batch_start, batch_end):
        if i < n:
            for c in range(c):
                for oy in range(out_h):
                    for ox in range(out_w):
                        sum_val = 0.0
                        for ky in range(kernel_size):
                            for kx in range(kernel_size):
                                iy = oy * stride + ky - padding
                                ix = ox * stride + kx - padding
                                if 0 <= iy < h and 0 <= ix < w:
                                    sum_val += tl.load(input_ptr + i * c * h * w + c * h * w + iy * w + ix)
                        avg_val = sum_val / (kernel_size * kernel_size)
                        tl.store(output_ptr + i * c * out_h * out_w + c * out_h * out_w + oy * out_w + ox, avg_val)

import torch

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size

    # Compute cosine similarity
    n1, d1 = x1.shape
    n2, d2 = x2.shape
    assert n1 == n2 and d1 == d2, "Input tensors must have the same shape"
    output = torch.empty((n1, 1), device=x1.device, dtype=x1.dtype)
    cosine_similarity_kernel[(n1,)](x1, x2, output, n1, n2, d1, x1.stride(0), x2.stride(0), output.stride(0), eps, BLOCK_SIZE=1024)

    # Add singleton dimension
    output = output.unsqueeze(2).unsqueeze(3)  # Shape: (n1, 1, 1, 1)

    # Apply 2D average pooling
    n, c, h, w = output.shape
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    pooled_output = torch.empty((n, c, out_h, out_w), device=x1.device, dtype=x1.dtype)
    avg_pool2d_kernel[(n,)](output, pooled_output, n, c, h, w, out_h, out_w, kernel_size, stride, padding, BLOCK_SIZE=1024)

    return pooled_output

import torch

# Example inputs
x1 = torch.randn(4, 3, device='cuda')
x2 = torch.randn(4, 3, device='cuda')
kernel_size = 2
stride = 1
padding = 0
eps = 1e-8

# Call the fused function
output = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride, padding, eps)

# Print the output
print(output)
