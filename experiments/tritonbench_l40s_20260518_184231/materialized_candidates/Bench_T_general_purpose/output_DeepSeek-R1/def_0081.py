import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    B, C, H, W, OH, OW,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < B * C * OH * OW
    
    bc = idx // (OH * OW)
    b = bc // C
    c = bc % C
    oh_ow = idx % (OH * OW)
    oh = oh_ow // OW
    ow = oh_ow % OW
    
    h_start = (oh * H) // OH
    h_end = ((oh + 1) * H) // OH
    w_start = (ow * W) // OW
    w_end = ((ow + 1) * W) // OW
    
    sum_val = 0.0
    num_elements = (h_end - h_start) * (w_end - w_start)
    
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_idx = (b * input_batch_stride) + (c * input_channel_stride) + (h * input_height_stride) + (w * input_width_stride)
            val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
            sum_val += val
    
    avg = sum_val / num_elements if num_elements > 0 else 0.0
    sigmoid_avg = 1.0 / (1.0 + tl.exp(-avg))
    
    output_idx = (b * output_batch_stride) + (c * output_channel_stride) + (oh * output_height_stride) + (ow * output_width_stride)
    tl.store(output_ptr + output_idx, sigmoid_avg, mask=mask)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if isinstance(output_size, int):
        OH = OW = output_size
    else:
        OH, OW = output_size
    B, C, H, W = input.shape
    output = torch.empty((B, C, OH, OW), device=input.device, dtype=input.dtype)
    
    grid = lambda meta: (triton.cdiv(B * C * OH * OW, meta['BLOCK_SIZE']), )
    
    sigmoid_adaptive_avg_pool2d_kernel[grid](
        input.data_ptr(), output.data_ptr(),
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        B, C, H, W, OH, OW,
        BLOCK_SIZE=1024
    )
    
    return output
