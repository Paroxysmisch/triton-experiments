import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    X,  # Input tensor
    Y,  # Output tensor
    output_size_h,  # Output height
    output_size_w,  # Output width
    input_size_h,  # Input height
    input_size_w,  # Input width
    stride_h,  # Stride height
    stride_w,  # Stride width
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + tl.arange(0, BLOCK_SIZE)
    y_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + tl.arange(0, BLOCK_SIZE)

    for i in range(output_size_h):
        for j in range(output_size_w):
            start_h = i * stride_h
            start_w = j * stride_w
            end_h = min(start_h + stride_h, input_size_h)
            end_w = min(start_w + stride_w, input_size_w)

            sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            count = 0

            for k in range(start_h, end_h):
                for l in range(start_w, end_w):
                    x_idx = x_offsets + k * input_size_w + l
                    sum_val += tl.load(X + x_idx, mask=mask)
                    count += 1

            avg_val = sum_val / count
            y_idx = y_offsets + i * output_size_w + j
            tl.store(Y + y_idx, avg_val, mask=mask)

import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(
    X1,  # First input tensor
    X2,  # Second input tensor
    D,  # Output distance tensor
    p,  # Norm degree
    eps,  # Small value to avoid division by zero
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X1.shape[0]

    x1_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + tl.arange(0, BLOCK_SIZE)
    x2_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + tl.arange(0, BLOCK_SIZE)

    diff = tl.abs(tl.load(X1 + x1_offsets, mask=mask) - tl.load(X2 + x2_offsets, mask=mask)) + eps
    dist = tl.sum(tl.pow(diff, p), axis=0)
    dist = tl.pow(dist, 1.0 / p)

    tl.store(D + offsets, dist, mask=mask)

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Ensure output_size is a tuple
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Adaptive average pooling
    pooled_x1 = torch.zeros((x1.shape[0], x1.shape[1], output_size[0], output_size[1]), device=x1.device)
    pooled_x2 = torch.zeros((x2.shape[0], x2.shape[1], output_size[0], output_size[1]), device=x2.device)

    stride_h = x1.shape[2] // output_size[0]
    stride_w = x1.shape[3] // output_size[1]

    adaptive_avg_pool2d_kernel[(x1.shape[0] * x1.shape[1],)](
        x1, pooled_x1, output_size[0], output_size[1], x1.shape[2], x1.shape[3], stride_h, stride_w, BLOCK_SIZE=1024
    )

    adaptive_avg_pool2d_kernel[(x2.shape[0] * x2.shape[1],)](
        x2, pooled_x2, output_size[0], output_size[1], x2.shape[2], x2.shape[3], stride_h, stride_w, BLOCK_SIZE=1024
    )

    # Pairwise distance calculation
    distance = torch.zeros((pooled_x1.shape[0],), device=x1.device)

    pairwise_distance_kernel[(pooled_x1.shape[0],)](
        pooled_x1, pooled_x2, distance, p, eps, BLOCK_SIZE=1024
    )

    if keepdim:
        distance = distance.unsqueeze(1)

    return distance
