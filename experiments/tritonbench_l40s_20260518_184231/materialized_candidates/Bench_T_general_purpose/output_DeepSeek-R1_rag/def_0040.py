import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    training, momentum, eps,
    S: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_c = tl.program_id(axis=0)
    if pid_c >= C:
        return

    running_mean = tl.load(running_mean_ptr + pid_c)
    running_var = tl.load(running_var_ptr + pid_c)

    if training:
        sum = 0.0
        sum_sq = 0.0
        for i in range(0, S, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            mask = idx < S
            ptr = input_ptr + idx * C + pid_c
            x = tl.load(ptr, mask=mask, other=0.0)
            sum += tl.sum(x)
            sum_sq += tl.sum(x * x)
        mean = sum / S
        var = (sum_sq / S) - (mean * mean)
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var
        tl.store(running_mean_ptr + pid_c, running_mean)
        tl.store(running_var_ptr + pid_c, running_var)
    else:
        mean = running_mean
        var = running_var

    inv_std = 1.0 / tl.sqrt(var + eps)
    gamma = tl.load(weight_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)

    for i in range(0, S, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < S
        ptr = input_ptr + idx * C + pid_c
        x = tl.load(ptr, mask=mask, other=0.0)
        x_hat = (x - mean) * inv_std
        y = gamma * x_hat + beta
        y_sigmoid = 1.0 / (1.0 + tl.exp(-y))
        tl.store(output_ptr + idx * C + pid_c, y_sigmoid, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    assert input.dim() in (2, 3), "Input must be 2D or 3D"
    N, C = input.shape[0], input.shape[1]
    L = input.shape[2] if input.dim() == 3 else 1
    S = N * L

    input_2d = input.contiguous().view(S, C)
    output = torch.empty_like(input_2d)

    weight = torch.ones(C, device=input.device, dtype=input.dtype) if weight is None else weight
    bias = torch.zeros(C, device=input.device, dtype=input.dtype) if bias is None else bias

    assert running_mean.shape == (C,), "running_mean must have shape (C,)"
    assert running_var.shape == (C,), "running_var must have shape (C,)"
    assert weight.shape == (C,), "weight must have shape (C,)"
    assert bias.shape == (C,), "bias must have shape (C,)"

    grid = (C,)
    BLOCK_SIZE = 1024
    sigmoid_batch_norm_kernel[grid](
        input_2d, running_mean, running_var, weight, bias, output,
        training, momentum, eps, S, C, BLOCK_SIZE=BLOCK_SIZE
    )

    return output.view(input.shape)
