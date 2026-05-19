import triton
import triton.language as tl

@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr, input_shape, output_shape, input_stride, output_stride,
    BLOCK_SIZE: tl.constexpr
):
    batch, in_channels, in_height, in_width = input_shape
    out_height, out_width = output_shape

    pid = tl.program_id(axis=0)
    num_elements = batch * in_channels * out_height * out_width
    pid = pid % num_elements

    b = pid // (in_channels * out_height * out_width)
    c = (pid % (in_channels * out_height * out_width)) // (out_height * out_width)
    oh = (pid % (out_height * out_width)) // out_width
    ow = pid % out_width

    ih = (in_height * oh) // out_height
    iw = (in_width * ow) // out_width
    h_start = (in_height * oh) // out_height
    h_end = (in_height * (oh + 1)) // out_height
    w_start = (in_width * ow) // out_width
    w_end = (in_width * (ow + 1)) // out_width

    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0

    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_idx = b * input_stride[0] + c * input_stride[1] + h * input_stride[2] + w * input_stride[3]
            sum_val += tl.load(input_ptr + input_idx)
            count += 1

    avg_val = sum_val / count
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))

    output_idx = b * output_stride[0] + c * output_stride[1] + oh * output_stride[2] + ow * output_stride[3]
    tl.store(output_ptr + output_idx, sigmoid_val)

import torch
import triton
import triton.language as tl

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    batch, in_channels, in_height, in_width = input.shape
    out_height, out_width = output_size

    output = torch.empty((batch, in_channels, out_height, out_width), device=input.device, dtype=input.dtype)

    input_ptr = input.contiguous().data_ptr()
    output_ptr = output.contiguous().data_ptr()

    input_shape = (batch, in_channels, in_height, in_width)
    output_shape = (out_height, out_width)

    input_stride = input.stride()
    output_stride = output.stride()

    grid = (batch * in_channels * out_height * out_width,)

    sigmoid_adaptive_avg_pool2d_kernel[grid](
        input_ptr, output_ptr, input_shape, output_shape, input_stride, output_stride,
        BLOCK_SIZE=1
    )

    return output
