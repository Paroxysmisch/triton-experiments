import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    primals_3_ptr,  # pointer to the input tensor (S, D)
    buf0_ptr,       # pointer to the mean buffer
    buf3_ptr,       # pointer to the auxiliary buffer
    buf4_ptr,       # pointer to the output tensor
    S, D,           # dimensions of the input tensor
    eps: tl.float32, # small epsilon value to avoid division by zero
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < S

    # Load the input tensor
    x = tl.load(primals_3_ptr + offsets * D, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=1) / D
    tl.store(buf0_ptr + offsets, mean, mask=mask)

    # Compute variance
    x_minus_mean = x - mean[:, None]
    var = tl.sum(x_minus_mean * x_minus_mean, axis=1) / D
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize the input tensor
    normalized = x_minus_mean * inv_std[:, None]
    tl.store(buf4_ptr + offsets * D, normalized, mask=mask)

    # Store auxiliary buffer (optional, depending on the use case)
    tl.store(buf3_ptr + offsets * D, inv_std[:, None], mask=mask)

import torch
import triton
import triton.language as tl

def fused_native_layer_norm(primals_1, primals_2, primals_3, eps=1e-5):
    S, D = primals_3.shape

    # Allocate buffers
    buf0 = torch.empty((S,), dtype=torch.float32, device=primals_3.device)
    buf3 = torch.empty((S, D), dtype=torch.float32, device=primals_3.device)
    buf4 = torch.empty((S, D), dtype=torch.float32, device=primals_3.device)

    # Define the grid and block size
    BLOCK_SIZE = 128
    grid = (S + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    triton_red_fused_native_layer_norm_0[grid, BLOCK_SIZE](
        primals_3, buf0, buf3, buf4, S, D, eps
    )

    return buf4, primals_3, buf0, buf3
