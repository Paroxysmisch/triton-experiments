import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(x1_ptr, x2_ptr, dist_ptr, N, D, p: tl.constexpr, eps: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    x1_row_ptr = x1_ptr + row_idx * D
    x2_row_ptr = x2_ptr + col_idx * D

    acc = tl.zeros((1,), dtype=tl.float32)

    for d in range(0, D):
        x1_val = tl.load(x1_row_ptr + d)
        x2_val = tl.load(x2_row_ptr + d)
        diff = x1_val - x2_val
        acc += tl.abs(diff) ** p

    distance = (acc + eps) ** (1.0 / p)
    tl.store(dist_ptr + row_idx * N + col_idx, distance)

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Apply adaptive average pooling
    pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
    pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)

    # Reshape pooled outputs to (N, D) where N is batch size and D is flattened spatial dimensions
    N1, C1, H1, W1 = pooled_x1.shape
    N2, C2, H2, W2 = pooled_x2.shape
    assert C1 == C2, "Channels of x1 and x2 must match after pooling"
    D = C1 * H1 * W1

    pooled_x1 = pooled_x1.view(N1, D)
    pooled_x2 = pooled_x2.view(N2, D)

    # Allocate output tensor
    distances = torch.empty((N1, N2), device=x1.device, dtype=torch.float32)

    # Launch Triton kernel
    grid = (N1, N2)
    pairwise_distance_kernel[grid](
        pooled_x1, pooled_x2, distances, N2, D, p, eps
    )

    # Optionally retain the reduced dimension
    if keepdim:
        distances = distances.unsqueeze(-1)

    return distances
