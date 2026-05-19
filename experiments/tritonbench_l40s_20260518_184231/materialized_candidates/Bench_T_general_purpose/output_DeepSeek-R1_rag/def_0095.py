import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    num_channels,
    num_elements,
    momentum,
    eps,
    training,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= num_channels:
        return

    mean = 0.0
    var = 0.0

    if training:
        sum_ = 0.0
        sum_sq = 0.0
        for i in range(0, num_elements, BLOCK_SIZE):
            off = pid * num_elements + i
            offsets = off + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (pid + 1) * num_elements
            x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            sum_ += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)
        mean = sum_ / num_elements
        var = (sum_sq / num_elements) - (mean * mean)

        running_mean = tl.load(running_mean_ptr + pid)
        running_var = tl.load(running_var_ptr + pid)
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var
        tl.store(running_mean_ptr + pid, running_mean)
        tl.store(running_var_ptr + pid, running_var)
    else:
        mean = tl.load(running_mean_ptr + pid)
        var = tl.load(running_var_ptr + pid)

    std = tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + pid) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + pid) if bias_ptr is not None else 0.0

    for i in range(0, num_elements, BLOCK_SIZE):
        off = pid * num_elements + i
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (pid + 1) * num_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        x_hat = (x - mean) / std
        x_hat = x_hat * weight + bias
        tl.store(output_ptr + offsets, x_hat, mask=mask)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05) -> torch.Tensor:
    original_shape = input.shape
    C = input.size(1)
    num_elements = input.numel() // C

    input_reshaped = input.permute(1, 0, *range(2, input.dim())).contiguous().view(C, -1)
    output = torch.empty_like(input_reshaped)

    if weight is None:
        weight = torch.ones(C, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = torch.zeros(C, dtype=input.dtype, device=input.device)

    running_mean = running_mean.to(device=input.device, dtype=input.dtype)
    running_var = running_var.to(device=input.device, dtype=input.dtype)

    grid = (C,)
    BLOCK_SIZE = 1024
    batch_norm_kernel[grid](
        input_reshaped,
        output,
        running_mean,
        running_var,
        weight,
        bias,
        num_channels=C,
        num_elements=num_elements,
        momentum=momentum,
        eps=eps,
        training=training,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output = output.view(original_shape[1], original_shape[0], *original_shape[2:])
    output = output.permute(1, 0, *range(2, output.dim())).contiguous()
    return output
