import triton
import triton.language as tl
import torch

@triton.jit
def batch_norm_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
                      num_features, num_batches, eps, training, momentum,
                      BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)

    # Compute the start index for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < num_features

    # Load data from memory
    input = tl.load(input_ptr + offsets, mask=mask)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask)
    running_var = tl.load(running_var_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask) if weight_ptr else 1.0
    bias = tl.load(bias_ptr + offsets, mask=mask) if bias_ptr else 0.0

    # Compute batch mean and variance if in training mode
    if training:
        mean = tl.sum(input, axis=0) / num_batches
        var = tl.sum((input - mean) ** 2, axis=0) / num_batches

        # Update running mean and variance
        running_mean = momentum * mean + (1 - momentum) * running_mean
        running_var = momentum * var + (1 - momentum) * running_var

    else:
        mean = running_mean
        var = running_var

    # Normalize
    normalized = (input - mean) / tl.sqrt(var + eps)
    output = normalized * weight + bias

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is a 2D tensor (batch_size, num_features)
    assert input.ndim == 2, "Input tensor must be 2D"
    batch_size, num_features = input.shape

    # Allocate output tensor
    output = torch.empty_like(input)

    # Convert optional parameters to pointers
    weight_ptr = weight if weight is not None else None
    bias_ptr = bias if bias is not None else None

    # Define grid size
    grid = lambda meta: (triton.cdiv(num_features, meta['BLOCK_SIZE']),)

    # Launch the kernel
    batch_norm_kernel[grid](
        input, running_mean, running_var, weight_ptr, bias_ptr, output,
        num_features, batch_size, eps, training, momentum,
        BLOCK_SIZE=1024
    )

    return output
