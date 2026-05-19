import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr, running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr, stride, N, C, H, W,
    eps: tl.float32, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input and parameters
    input_offsets = tl.arange(0, N)[:, None, None, None] * stride + \
                    tl.arange(0, C)[None, :, None, None] * H * W + \
                    tl.arange(0, H)[None, None, :, None] * W + \
                    offsets[None, None, None, :]
    input = tl.load(input_ptr + input_offsets, mask=offsets < N * C * H * W, other=0.0)

    running_mean = tl.load(running_mean_ptr + tl.arange(0, C))
    running_var = tl.load(running_var_ptr + tl.arange(0, C))

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + tl.arange(0, C))
    else:
        weight = tl.full((C,), 1.0, dtype=tl.float32)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, C))
    else:
        bias = tl.full((C,), 0.0, dtype=tl.float32)

    # Normalize
    normalized = (input - running_mean[:, None, None]) / tl.sqrt(running_var[:, None, None] + eps)
    output = normalized * weight[:, None, None] + bias[:, None, None]

    # Store output
    tl.store(output_ptr + input_offsets, output, mask=offsets < N * C * H * W)

import torch
import triton
import triton.language as tl

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    N, C, H, W = input.shape
    stride = C * H * W

    # Convert to Triton-compatible tensors
    input_ptr = input.contiguous().to(torch.float32).data_ptr()
    running_mean_ptr = running_mean.contiguous().to(torch.float32).data_ptr()
    running_var_ptr = running_var.contiguous().to(torch.float32).data_ptr()

    if weight is not None:
        weight_ptr = weight.contiguous().to(torch.float32).data_ptr()
    else:
        weight_ptr = None

    if bias is not None:
        bias_ptr = bias.contiguous().to(torch.float32).data_ptr()
    else:
        bias_ptr = None

    output = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (N * C * H * W // 1024 + 1,)
    batch_norm_kernel[grid](
        input_ptr, output.data_ptr(), running_mean_ptr, running_var_ptr,
        weight_ptr, bias_ptr, stride, N, C, H, W, eps, BLOCK_SIZE=1024
    )

    return output
