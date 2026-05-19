import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(X_ptr, Y_ptr, W_ptr, B_ptr, O_ptr, 
                                   num_channels, num_groups, eps,
                                   BLOCK_SIZE: tl.constexpr):
    # Compute the index of the current block
    pid = tl.program_id(0)
    # Create pointers for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load input tensors
    X = tl.load(X_ptr + offsets, mask=offsets < num_channels, other=0)
    Y = tl.load(Y_ptr + offsets, mask=offsets < num_channels, other=0)
    # Perform element-wise addition
    Z = X + Y
    # Perform element-wise multiplication
    M = Z * Y
    # Calculate the group size
    group_size = num_channels // num_groups
    # Calculate group indices
    group_idx = offsets // group_size
    # Compute mean and variance for group normalization
    group_mean = tl.sum(M, axis=0) / group_size
    group_var = tl.sum((M - group_mean) ** 2, axis=0) / group_size
    # Normalize
    M_normalized = (M - group_mean) / tl.sqrt(group_var + eps)
    # Load weight and bias
    W = tl.load(W_ptr + offsets, mask=offsets < num_channels, other=1)
    B = tl.load(B_ptr + offsets, mask=offsets < num_channels, other=0)
    # Apply affine transformation
    O = M_normalized * W + B
    # Store the result
    tl.store(O_ptr + offsets, O, mask=offsets < num_channels)

import torch

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    assert input1.shape == input2.shape, "Input tensors must be broadcastable to each other."
    C = input1.shape[1]  # Assuming input is in NCHW format
    assert weight.shape == (C,), "Weight tensor must have shape (C,)."
    assert bias.shape == (C,), "Bias tensor must have shape (C,)."
    assert C % num_groups == 0, "Number of channels must be divisible by num_groups."

    # Flatten input tensors for the Triton kernel
    X_flat = input1.flatten()
    Y_flat = input2.flatten()
    W_flat = weight.flatten()
    B_flat = bias.flatten()

    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input1)

    # Launch the Triton kernel
    num_channels = X_flat.numel()
    BLOCK_SIZE = 1024  # Define block size based on your GPU architecture
    grid = (num_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_add_mul_groupnorm_kernel[grid](
        X_flat, Y_flat, W_flat, B_flat, out.flatten(),
        num_channels, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
