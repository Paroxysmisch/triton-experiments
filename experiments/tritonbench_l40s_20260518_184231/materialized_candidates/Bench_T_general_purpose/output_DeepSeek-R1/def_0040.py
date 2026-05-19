import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr, output_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, L,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C * L
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Convert linear index to (n, c, l)
    l = offsets % L
    c = (offsets // L) % C
    n = offsets // (C * L)

    # Load input
    input_offsets = n * C * L + c * L + l
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # Load parameters for the channel
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # Compute normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (x - mean) * inv_std
    scaled = normalized * weight + bias

    # Apply sigmoid
    sigmoid = 1.0 / (1.0 + tl.exp(-scaled))

    # Store output
    tl.store(output_ptr + input_offsets, sigmoid, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is contiguous
    input = input.contiguous()
    input_3d = input.dim() == 3
    if input.dim() == 2:
        input = input.unsqueeze(2)  # (N, C, 1)
    N, C, L = input.shape

    # Compute current mean and variance
    if training:
        # Calculate mean and var over (N, L)
        current_mean = input.mean(dim=(0, 2))
        current_var = input.var(dim=(0, 2), unbiased=False)
        # Update running stats
        running_mean.copy_((1 - momentum) * running_mean + momentum * current_mean.detach())
        running_var.copy_((1 - momentum) * running_var + momentum * current_var.detach())
    else:
        current_mean = running_mean
        current_var = running_var

    # Handle optional weight and bias
    if weight is None:
        weight = torch.ones_like(current_mean, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = torch.zeros_like(current_mean, dtype=input.dtype, device=input.device)

    # Prepare output tensor
    output = torch.empty_like(input)

    # Launch kernel
    total_elements = N * C * L
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    sigmoid_batch_norm_kernel[grid](
        input, output, current_mean, current_var, weight, bias,
        N, C, L,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back if input was 2D
    if not input_3d:
        output = output.squeeze(2)

    return output
