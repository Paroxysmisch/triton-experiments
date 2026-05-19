import torch
import triton
import triton.language as tl
from ..utils import calculate_settings

@triton.autotune(
    configs=calculate_settings(num_warps=7),
    key=['kernel_size', 'in_channels', 'out_channels', 'batch', 'feat_size'],
)
@triton.jit
def pixel_shuffle_conv2d_triton(
    input_pointer, weight_pointer, bias_pointer, output_pointer,
    batch, feat_size, in_channels, out_channels, 
    input_stride_0, input_stride_1, 
    weight_stride_0, weight_stride_1, weight_stride_2, 
    bias_stride, 
    output_stride_0, output_stride_1, 
    stride, padding, kernel_size, upscale_factor, groups, 
    NUM_WARPS: tl.constexpr, 
    BLOCK_SIZE_BATCH_FEAT: tl.constexpr, 
    BLOCK_SIZE_CHANNEL: tl.constexpr, 
    BLOCK_SIZE_OUTPUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(feat_size, BLOCK_SIZE_BATCH_FEAT)
    num_pid_n = tl.cdiv(in_channels, BLOCK_SIZE_CHANNEL)
    num_pid_in_group = GROUP_SIZE_BATCH * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_BATCH
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_BATCH)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    input_m_offset = (pid_m * BLOCK_SIZE_BATCH_FEAT + tl.arange(0, BLOCK_SIZE_BATCH_FEAT))[:, None]
    input_n_offset = (pid_n * BLOCK_SIZE_CHANNEL + tl.arange(0, BLOCK_SIZE_CHANNEL))[None, :]
    input_offset = input_m_offset * input_stride_0 + input_n_offset * input_stride_1

    weight_m_offset = tl.arange(0, BLOCK_SIZE_OUTPUT)[:, None]
    weight_n_offset = tl.arange(0, BLOCK_SIZE_CHANNEL)[None, :]
    weight_offset = weight_m_offset * weight_stride_0 + weight_n_offset * weight_stride_1

    bias_offset = tl.arange(0, BLOCK_SIZE_OUTPUT)

    output_m_offset = (pid_m * BLOCK_SIZE_BATCH_FEAT + tl.arange(0, BLOCK_SIZE_BATCH_FEAT) // upscale_factor)[:, None]
    output_n_offset = (pid_n * BLOCK_SIZE_CHANNEL + tl.arange(0, BLOCK_SIZE_CHANNEL))[None, :] // upscale_factor
    output_offset = output_m_offset * output_stride_0 + output_n_offset * output_stride_1

    accumulator = tl.zeros((BLOCK_SIZE_BATCH_FEAT, BLOCK_SIZE_OUTPUT), dtype=tl.float32)
    for k in range(tl.cdiv(kernel_size, BLOCK_SIZE_CHANNEL)):
        input_pointer_offset = input_pointer + input_offset + k * weight_stride_2
        weight_pointer_offset = weight_pointer + weight_offset + k * BLOCK_SIZE_CHANNEL * BLOCK_SIZE_CHANNEL
        input = tl.load(input_pointer_offset, mask=(input_m_offset < feat_size) & (input_n_offset < in_channels), other=0.0)
        weight = tl.load(weight_pointer_offset, mask=(weight_m_offset < kernel_size) & (weight_n_offset < in_channels), other=0.0)
        accumulator += tl.dot(input, weight)

    if bias_pointer is not None:
        bias_pointer_offset = bias_pointer + bias_offset
        bias = tl.load(bias_pointer_offset, mask=bias_offset < out_channels, other=0.0)
        accumulator += bias

    output_pointer_offset = output_pointer + output_offset
    tl.store(output_pointer_offset, accumulator, mask=(output_m_offset < feat_size) & (output_n_offset < in_channels))

def pixel_shuffle_conv2d_triton_wrapper(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor = None, 
    stride: int = 1, 
    padding: int = 0, 
    upscale_factor: int = 2, 
    groups: int = 1
) -> torch.Tensor:
    batch, in_channels, feat_size, _ = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    assert (in_channels * groups == weight.shape[1] * weight.shape[2])
    assert (in_channels % groups == 0)
    assert (kernel_size == weight.shape[2])
    assert (kernel_size in {3, 5, 7})

    output = torch.empty((batch, out_channels, feat_size, feat_size), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(in_channels, META['BLOCK_SIZE_CHANNEL']) * triton.cdiv(feat_size, META['BLOCK_SIZE_BATCH_FEAT']), )
    pixel_shuffle_conv2d_triton[grid](
        x, weight, bias, output,
        batch, feat_size, in_channels, out_channels,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1), weight.stride(2),
        bias.stride(0) if bias is not None else 0,
        output.stride(0), output.stride(1),
        stride, padding, kernel_size, upscale_factor, groups
    )
    return output
