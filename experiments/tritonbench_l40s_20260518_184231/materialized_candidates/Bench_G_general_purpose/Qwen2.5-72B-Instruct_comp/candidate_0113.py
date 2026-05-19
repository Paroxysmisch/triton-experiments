import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3_ptr,  # Input tensor (S, D)
    primals_1_ptr,  # Scale tensor (D)
    primals_2_ptr,  # Bias tensor (D)
    out_ptr0,       # Output buffer for mean, variance, and count (3, D)
    out_ptr1,       # Output tensor (S, D)
    S, D,           # Dimensions of the input tensor
    RBLOCK: tl.constexpr,  # Number of elements processed by each thread block
    EPSILON: tl.constexpr  # Small value to avoid division by zero
):
    # Get the row index
    row_idx = tl.program_id(0)
    if row_idx >= S:
        return

    # Initialize shared memory for mean, variance, and count
    tmp3_mean = tl.zeros((1,), dtype=tl.float32)
    tmp3_m2 = tl.zeros((1,), dtype=tl.float32)
    tmp3_weight = tl.zeros((1,), dtype=tl.float32)

    # Process elements in chunks
    for offset in range(0, D, RBLOCK):
        col_idx = offset + tl.arange(0, RBLOCK)
        mask = col_idx < D
        x = tl.load(primals_3_ptr + row_idx * D + col_idx, mask=mask)

        # Welford algorithm for mean and variance
        delta = x - tmp3_mean
        tmp3_mean += delta / (tmp3_weight + 1)
        delta2 = x - tmp3_mean
        tmp3_m2 += delta * delta2
        tmp3_weight += 1

    # Store the mean, variance, and count in the output buffer
    tl.store(out_ptr0 + 0 * D + row_idx, tmp3_mean)
    tl.store(out_ptr0 + 1 * D + row_idx, tmp3_m2 / tmp3_weight)
    tl.store(out_ptr0 + 2 * D + row_idx, tmp3_weight)

    # Load the mean and variance
    mean = tl.load(out_ptr0 + 0 * D + row_idx)
    variance = tl.load(out_ptr0 + 1 * D + row_idx)
    count = tl.load(out_ptr0 + 2 * D + row_idx)

    # Compute the normalization factor
    norm_factor = 1.0 / tl.sqrt(variance / count + EPSILON)

    # Apply the normalization and optional affine transformations
    for col_idx in range(0, D):
        x = tl.load(primals_3_ptr + row_idx * D + col_idx)
        scale = tl.load(primals_1_ptr + col_idx)
        bias = tl.load(primals_2_ptr + col_idx)
        y = (x - mean) * norm_factor * scale + bias
        tl.store(out_ptr1 + row_idx * D + col_idx, y)

import torch
import triton
import triton.language as tl

def fused_native_layer_norm(primals_3, primals_1, primals_2, EPSILON=1e-5, RBLOCK=256):
    # Ensure the input tensors are on the same device
    device = primals_3.device
    S, D = primals_3.shape

    # Allocate output buffers
    out_ptr0 = torch.empty((3, D), device=device, dtype=torch.float32)
    out_ptr1 = torch.empty_like(primals_3)

    # Define the grid and block configuration
    grid = (S, 1, 1)
    block = (RBLOCK, 1, 1)

    # Launch the Triton kernel
    triton_red_fused_native_layer_norm_0[grid, block](
        primals_3, primals_1, primals_2, out_ptr0, out_ptr1, S, D, RBLOCK, EPSILON
    )

    # Return the normalized output and the mean, variance, and count
    return out_ptr1, out_ptr0

# Example usage
S, D = 1024, 1024
primals_3 = torch.randn(S, D, device='cuda')
primals_1 = torch.randn(D, device='cuda')
primals_2 = torch.randn(D, device='cuda')

normalized_output, stats = fused_native_layer_norm(primals_3, primals_1, primals_2)
print(normalized_output.shape)  # Should be (S, D)
print(stats.shape)  # Should be (3, D)
