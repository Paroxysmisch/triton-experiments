import triton
import triton.language as tl
import torch

@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr, running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr, N, C, H, W,
    momentum, eps, training, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE

    # Load data
    input = tl.load(input_ptr + offset)
    running_mean = tl.load(running_mean_ptr + offset)
    running_var = tl.load(running_var_ptr + offset)

    if training:
        # Compute mean and variance
        mean = tl.sum(input, axis=0) / N
        var = tl.sum((input - mean) ** 2, axis=0) / N

        # Update running mean and variance
        running_mean = momentum * mean + (1 - momentum) * running_mean
        running_var = momentum * var + (1 - momentum) * running_var

        # Store updated running mean and variance
        tl.store(running_mean_ptr + offset, running_mean)
        tl.store(running_var_ptr + offset, running_var)
    else:
        mean = running_mean
        var = running_var

    # Normalize
    normalized = (input - mean) / tl.sqrt(var + eps)

    # Apply scale and shift
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offset)
        normalized *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offset)
        normalized += bias

    # Store result
    tl.store(output_ptr + offset, normalized)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Input dimensions
    N, C, H, W = input.shape

    # Prepare output tensor
    output = torch.empty_like(input)

    # Convert all tensors to contiguous memory format for Triton
    input = input.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Launch Triton kernel
    BLOCK_SIZE = 1024  # You can adjust this based on your hardware capabilities
    grid = lambda meta: (triton.cdiv(N * C * H * W, meta['BLOCK_SIZE']),)
    batch_norm_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        N=N, C=C, H=H, W=W,
        momentum=momentum, eps=eps, training=training,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
