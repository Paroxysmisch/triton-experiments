import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, L, eps, weight_provided, bias_provided
):
    # Calculate the index for the current thread
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C

    # Compute the mean and variance for the current channel
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)

    # Compute the normalization factor
    norm_factor = tl.rsqrt(var + eps)

    # Load weight and bias if provided
    weight = tl.load(weight_ptr + channel_idx) if weight_provided else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_provided else 0.0

    # Iterate over the sequence length
    for i in range(L):
        idx = batch_idx * C * L + channel_idx * L + i
        # Load the input value
        input_val = tl.load(input_ptr + idx)
        # Apply batch normalization
        normalized = (input_val - mean) * norm_factor
        # Apply scale and shift
        scaled = normalized * weight + bias
        # Apply sigmoid activation
        sigmoid_output = 1.0 / (1.0 + tl.exp(-scaled))
        # Store the result
        tl.store(output_ptr + idx, sigmoid_output)

import torch

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is a 3D tensor for uniform processing
    if input.dim() == 2:
        input = input.unsqueeze(-1)
    N, C, L = input.shape

    # Prepare output tensor
    output = torch.empty_like(input)

    # Handle weight and bias
    weight_provided = weight is not None
    bias_provided = bias is not None

    if weight is None:
        weight = torch.ones(C, device=input.device, dtype=input.dtype)
    if bias is None:
        bias = torch.zeros(C, device=input.device, dtype=input.dtype)

    # Launch the Triton kernel
    grid = lambda meta: (N * C,)
    sigmoid_batch_norm_kernel[grid](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C=C, L=L,
        eps=eps,
        weight_provided=weight_provided,
        bias_provided=bias_provided
    )

    # If training, update running statistics
    if training:
        # Compute the new mean and variance
        batch_mean = input.mean(dim=(0, 2))
        batch_var = input.var(dim=(0, 2), unbiased=False)

        # Update running mean and variance
        running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
        running_var.mul_(1 - momentum).add_(momentum * batch_var)

    # Squeeze the output if the input was 2D
    if input.dim() == 2:
        output = output.squeeze(-1)

    return output
