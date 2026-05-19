import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    output_batch_stride,
    output_channel_stride,
    n_channels,
    training,
    momentum,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offs_c < n_channels

    var_scale = tl.load(running_var_ptr + offs_c, mask=mask_c)
    mean_shift = tl.load(running_mean_ptr + offs_c, mask=mask_c)

    var_eps = var_scale + eps
    norm_factor = tl.math.rsqrt(var_eps)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_input = (
        offs_n[:, None] * input_batch_stride
        + offs_c[None, :] * input_channel_stride
    )
    offs_output = (
        offs_n[:, None] * output_batch_stride
        + offs_c[None, :] * output_channel_stride
    )
    mask_n = offs_n < input_batch_stride

    if training:
        for i in range(0, tl.cdiv(n_channels, BLOCK_SIZE_C)):
            batch_norm_update_stats_kernel(
                input_ptr + offs_input,
                running_mean_ptr,
                running_var_ptr,
                input_batch_stride,
                input_channel_stride,
                n_channels,
                momentum,
                eps,
                BLOCK_SIZE_N,
                BLOCK_SIZE_C,
            )
            tl.debug_barrier()

        tl.device_assert(
            tl.all(mask_n), "batch dimension out of bounds: " + str(pid_n)
        )
        tl.device_assert(
            tl.all(mask_c), "channel dimension out of bounds: " + str(pid_c)
        )

        input_value = tl.load(
            input_ptr + offs_input, mask=mask_n[:, None] & mask_c[None, :]
        ).to(tl.float32)

        normed_value = (input_value - mean_shift) * norm_factor

        if weight_ptr is not None:
            weight = tl.load(weight_ptr + offs_c, mask=mask_c).to(tl.float32)
            normed_value = normed_value * weight[None, :]

        if bias_ptr is not None:
            bias = tl.load(bias_ptr + offs_c, mask=mask_c).to(tl.float32)
            normed_value = normed_value + bias[None, :]

        tl.store(
            output_ptr + offs_output,
            normed_value.to(input_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_c[None, :],
        )
    else:
        tl.device_assert(
            tl.all(mask_n), "batch dimension out of bounds: " + str(pid_n)
        )
        tl.device_assert(
            tl.all(mask_c), "channel dimension out of bounds: " + str(pid_c)
        )

        input_value = tl.load(
            input_ptr + offs_input, mask=mask_n[:, None] & mask_c[None, :]
        ).to(tl.float32)

        normed_value = (input_value - mean_shift) * norm_factor

        if weight_ptr is not None:
            weight = tl.load(weight_ptr + offs_c, mask=mask_c).to(tl.float32)
            normed_value = normed_value * weight[None, :]

        if bias_ptr is not None:
            bias = tl.load(bias_ptr + offs_c, mask=mask_c).to(tl.float32)
            normed_value = normed_value + bias[None, :]

        tl.store(
            output_ptr + offs_output,
            normed_value.to(input_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_c[None, :],
        )

@triton.jit
def batch_norm_update_stats_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    input_batch_stride,
    input_channel_stride,
    n_channels,
    momentum,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offs_c < n_channels

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_input = (
        offs_n[:, None] * input_batch_stride
        + offs_c[None, :] * input_channel_stride
    )
    mask_n = offs_n < input_batch_stride

    tl.device_assert(
        tl.all(mask_n), "batch dimension out of bounds: " + str(pid_n)
    )
    tl.device_assert(
        tl.all(mask_c), "channel dimension out of bounds: " + str(pid_c)
    )

    input_value = tl.load(
        input_ptr + offs_input, mask=mask_n[:, None] & mask_c[None, :]
    ).to(tl.float32)

    batch_mean = tl.sum(input_value, axis=0) / input_batch_stride
    batch_var = tl.sum((input_value - batch_mean[:, None]) ** 2, axis=0) / input_batch_stride

    running_var = tl.load(running_var_ptr + offs_c, mask=mask_c)
    running_mean = tl.load(running_mean_ptr + offs_c, mask=mask_c)

    running_var_new = momentum * running_var + (1 - momentum) * batch_var
    running_mean_new = momentum * running_mean + (1 - momentum) * batch_mean

    tl.store(running_var_ptr + offs_c, running_var_new, mask=mask_c)
    tl.store(running_mean_ptr + offs_c, running_mean_new, mask=mask_c)

def batch_norm(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: Optional[bool] = False,
    momentum: Optional[float] = 0.1,
    eps: Optional[float] = 1e-05
