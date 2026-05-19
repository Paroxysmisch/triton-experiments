import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for pairwise distance after adaptive average pooling
@triton.jit
def pairwise_distance_kernel(
    x1_ptr, x2_ptr, out_ptr,
    output_size_h, output_size_w, 
    n_elements, p, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Define a block of work
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load data from global memory into registers
    x1 = tl.load(x1_ptr + block_start, mask=block_start + tl.arange(0, BLOCK_SIZE) < n_elements, other=0.0)
    x2 = tl.load(x2_ptr + block_start, mask=block_start + tl.arange(0, BLOCK_SIZE) < n_elements, other=0.0)

    # Compute pairwise distance with norm p
    diff = x1 - x2
    distance = tl.pow(tl.abs(diff) + eps, p)
    result = tl.sum(distance)

    # Store result back to global memory
    tl.store(out_ptr + pid, result)

# Wrapper function
def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    # Apply adaptive average pooling to both input tensors
    x1_pooled = F.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = F.adaptive_avg_pool2d(x2, output_size)

    # Flatten the pooled tensors for pairwise distance calculation
    x1_flat = x1_pooled.view(x1_pooled.size(0), -1)
    x2_flat = x2_pooled.view(x2_pooled.size(0), -1)

    # Prepare output tensor
    output = torch.empty(x1_flat.size(0), device=x1.device)

    # Launch Triton kernel
    BLOCK_SIZE = 1024
    n_elements = x1_flat.size(1)
    grid = (x1_flat.size(0),)

    pairwise_distance_kernel[grid](
        x1_flat, x2_flat, output,
        output_size[0] if isinstance(output_size, tuple) else output_size,
        output_size[1] if isinstance(output_size, tuple) else output_size,
        n_elements, p, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Optionally keep the reduced dimension
    if keepdim:
        output = output.view(-1, 1)

    return output

# Example usage
x1 = torch.randn(10, 3, 32, 32, device='cuda')
x2 = torch.randn(10, 3, 32, 32, device='cuda')
output_size = (8, 8)
result = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size)
print(result)
