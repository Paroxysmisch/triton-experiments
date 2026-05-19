import triton
import triton.language as tl
import torch

@triton.jit
def batch_norm_sigmoid_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, L, eps, training, momentum, BLOCK_SIZE: tl.constexpr
):
    # Pointers to the input data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data
    input = tl.load(input_ptr + offsets)
    running_mean = tl.load(running_mean_ptr + offsets)
    running_var = tl.load(running_var_ptr + offsets)
    weight = tl.load(weight_ptr + offsets) if weight_ptr else 1.0
    bias = tl.load(bias_ptr + offsets) if bias_ptr else 0.0

    # Compute mean and variance
    if training:
        mean = tl.sum(input, axis=0) / N
        var = tl.sum((input - mean) ** 2, axis=0) / N
        # Update running statistics
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        tl.store(running_mean_ptr + offsets, running_mean)
        tl.store(running_var_ptr + offsets, running_var)
    else:
        mean = running_mean
        var = running_var

    # Normalize
    inv_std = tl.rsqrt(var + eps)
    norm_input = (input - mean) * inv_std * weight + bias

    # Apply sigmoid
    output = 1 / (1 + tl.exp(-norm_input))

    # Store output
    tl.store(output_ptr + offsets, output)


def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    N, C, *L = input.shape
    input_flat = input.view(N, C, -1)  # Flatten the sequence length if present
    numel = input_flat.numel()

    # Allocate output tensor
    output = torch.empty_like(input_flat)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, BLOCK_SIZE),)
    batch_norm_sigmoid_kernel[grid](
        input_flat,
        running_mean,
        running_var,
        weight if weight is not None else torch.ones(C, device=input.device),
        bias if bias is not None else torch.zeros(C, device=input.device),
        output,
        N, C, L[0] if L else 1,
        eps, training, momentum,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output.view_as(input)
