import torch
import triton
import triton.language as tl

# Triton kernel for computing cosine similarity
@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr, n, d, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    x1_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x2_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_x1 = x1_idx < n
    mask_x2 = x2_idx < n

    x1_norm = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    x2_norm = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dot_product = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute norms and dot product
    for i in range(d):
        x1_val = tl.load(x1_ptr + i * n + x1_idx, mask=mask_x1)
        x2_val = tl.load(x2_ptr + i * n + x2_idx, mask=mask_x2)
        x1_norm += x1_val * x1_val
        x2_norm += x2_val * x2_val
        dot_product += x1_val * x2_val

    # Normalize and compute cosine similarity
    x1_norm = tl.sqrt(x1_norm)
    x2_norm = tl.sqrt(x2_norm)
    cos_sim = dot_product / (x1_norm * x2_norm + eps)

    tl.store(out_ptr + x1_idx, cos_sim, mask=mask_x1)

# Wrapper function for fused_avg_pool2d_cosine_similarity
def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    if stride is None:
        stride = kernel_size

    # Ensure inputs are on the same device
    device = x1.device
    assert device.type == "cuda", "Inputs must be on a CUDA device"
    assert x1.device == x2.device, "Inputs must be on the same device"

    n, d = x1.shape
    B = 1  # Batch size of 1 since we're processing one pair at a time
    H, W = 1, 1  # Height and width of the feature map

    # Pad the inputs if needed
    padded_n = n + 2 * padding
    padded_x1 = torch.nn.functional.pad(x1, (padding, padding))
    padded_x2 = torch.nn.functional.pad(x2, (padding, padding))

    # Compute cosine similarity
    cos_sim_out = torch.empty((n,), device=device)
    cosine_similarity_kernel[grid=(n // BLOCK_SIZE + 1, 1), block=(BLOCK_SIZE,)](
        padded_x1.data_ptr(), padded_x2.data_ptr(), cos_sim_out.data_ptr(), padded_n, d
    )

    # Add a singleton dimension
    cos_sim_out = cos_sim_out.unsqueeze(-1)

    # Reshape for avg_pool2d
    cos_sim_out = cos_sim_out.view(B, H, W, n, 1)

    # Apply 2D average pooling
    pool_out = torch.nn.functional.avg_pool2d(cos_sim_out, kernel_size, stride=stride, padding=0)

    return pool_out.squeeze()

# Example usage
x1 = torch.randn(10, 10, device="cuda")
x2 = torch.randn(10, 10, device="cuda")
result = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2)
print(result)
