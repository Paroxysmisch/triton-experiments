import triton
import triton.language as tl
import torch

@triton.jit
def fused_add_mul_groupnorm_kernel(
    X_ptr, Y_ptr, weight_ptr, bias_ptr, out_ptr,
    C, num_groups, eps, stride, BLOCK_SIZE: tl.constexpr
):
    # Program ID corresponds to the batch index
    pid = tl.program_id(axis=0)

    # Compute the offset for the current batch
    offset = pid * stride

    # Load data from memory
    X = tl.load(X_ptr + offset + tl.arange(0, BLOCK_SIZE))
    Y = tl.load(Y_ptr + offset + tl.arange(0, BLOCK_SIZE))

    # Element-wise addition
    Z = X + Y

    # Element-wise multiplication
    M = Z * Y

    # Group normalization
    # Calculate mean and variance for each group
    group_size = C // num_groups
    group_start = tl.arange(0, BLOCK_SIZE) // group_size * group_size
    group_mean = tl.sum(M, axis=0) / group_size
    group_var = tl.sum((M - group_mean) ** 2, axis=0) / group_size
    inv_std = tl.rsqrt(group_var + eps)

    # Apply normalization
    normalized = (M - group_mean) * inv_std

    # Apply affine transformation
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE))
    O = normalized * weight + bias

    # Store result
    tl.store(out_ptr + offset + tl.arange(0, BLOCK_SIZE), O)


def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    assert input1.shape == input2.shape, "Input tensors must have the same shape"
    C = input1.shape[1]  # Assuming input is in NCHW format
    assert weight.shape[0] == C and bias.shape[0] == C, "Weight and bias must have shape (C,)"
    assert C % num_groups == 0, "num_groups must divide the number of channels C evenly"

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input1)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define a suitable block size
    grid = (input1.shape[0],)  # One block per batch

    fused_add_mul_groupnorm_kernel[grid](
        input1, input2, weight, bias, out,
        C, num_groups, eps, input1.stride(0), BLOCK_SIZE=BLOCK_SIZE
    )

    return out
