import torch
import triton
import triton.language as tl

@triton.jit
def _apply_bn_kernel(
    x_ptr,           # Pointer to input tensor data (flattened).
    y_ptr,           # Pointer to output tensor data (flattened).
    mean_ptr,        # Pointer to per-channel mean.
    var_ptr,         # Pointer to per-channel variance.
    weight_ptr,      # Pointer to per-channel weight (gamma), if any.
    bias_ptr,        # Pointer to per-channel bias (beta), if any.
    total_rows,      # N * spatial_dims, total rows for each channel.
    C,               # Number of channels.
    eps,             # Epsilon for numerical stability.
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (total_rows * C)

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Identify channel index
    c = offsets % C

    # Load mean and variance for this channel
    mean_val = tl.load(mean_ptr + c, mask=(c < C))
    var_val = tl.load(var_ptr + c, mask=(c < C))

    # Compute BN: (x - mean) / sqrt(var + eps)
    normed = (x - mean_val) / tl.sqrt(var_val + eps)

    # Optionally apply weight and bias
    if has_weight:
        w_val = tl.load(weight_ptr + c, mask=(c < C))
        normed = normed * w_val
    if has_bias:
        b_val = tl.load(bias_ptr + c, mask=(c < C))
        normed = normed + b_val

    # Store result
    tl.store(y_ptr + offsets, normed, mask=mask)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    """
    Applies Batch Normalization for each channel across a batch of data.

    input:        (N, C, ...) shaped tensor.
    running_mean: 1D tensor of shape (C,).
    running_var:  1D tensor of shape (C,).
    weight:       Optional 1D tensor of shape (C,).
    bias:         Optional 1D tensor of shape (C,).
    training:     Bool indicating if the layer is in training mode.
    momentum:     Value used for the running_mean and running_var computation.
    eps:          A value added to the denominator for numerical stability.
    """
    # Check at least 2D
    if input.dim() < 2:
        raise ValueError("batch_norm expects input with at least 2 dimensions (N, C, ...)")

    # Dimensions
    N, C = input.shape[0], input.shape[1]
    # Flatten everything except channel into one dimension
    # shape becomes (N, C, remaining_dims)
    flat_input = input.view(N, C, -1)

    # Compute mean/var if training
    if training:
        # Mean and var across batch and spatial dims
        mean = flat_input.mean(dim=[0, 2])
        var = flat_input.var(dim=[0, 2], unbiased=False)
        # Update running stats
        running_mean.data = (1.0 - momentum) * running_mean.data + momentum * mean.data
        running_var.data = (1.0 - momentum) * running_var.data + momentum * var.data
        used_mean = mean
        used_var = var
    else:
        # Inference mode: use running stats
        used_mean = running_mean
        used_var = running_var

    # Prepare output
    output = torch.empty_like(input)
    flat_output = output.view(N, C, -1)

    # Total elements to process is N*C*spatial
    total_elements = flat_output.numel()

    # Launch config
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    # Launch BN kernel
    _apply_bn_kernel[grid](
        flat_input, 
        flat_output,
        used_mean, 
        used_var,
        weight if weight is not None else 0, 
        bias if bias is not None else 0,
        N * flat_input.shape[2],  # total rows for each channel
        C,
        eps,
        has_weight=(weight is not None),
        has_bias=(bias is not None),
        BLOCK_SIZE=1024
    )

    return output
