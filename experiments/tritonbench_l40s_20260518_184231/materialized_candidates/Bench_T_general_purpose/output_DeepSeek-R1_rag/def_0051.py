import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    B,
    C,
    iW,
    oW,
    kernel_size,
    stride,
    padding,
    count_include_pad,
    BLOCK_SIZE: tl.constexpr,
):
    batch_channel = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    oW_idx = block_start + offsets

    batch_id = batch_channel // C
    channel_id = batch_channel % C

    batch_mask = (batch_id < B) & (channel_id < C)
    mask = (oW_idx < oW) & batch_mask

    input_base = batch_id * C * iW + channel_id * iW
    output_base = batch_id * C * oW + channel_id * oW

    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    valid_count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for k in range(kernel_size):
        input_pos = (oW_idx * stride - padding) + k
        valid_input = (input_pos >= 0) & (input_pos < iW)
        input_val = tl.load(
            input_ptr + input_base + input_pos,
            mask=valid_input & mask,
            other=0.0,
        )
        sum += input_val
        if count_include_pad:
            valid_count += 1
        else:
            valid_count += tl.where(valid_input, 1, 0)

    divisor = tl.where(count_include_pad, kernel_size, valid_count)
    divisor_safe = tl.maximum(divisor, 1)
    avg = sum / divisor_safe

    tl.store(output_ptr + output_base + oW_idx, avg, mask=mask)

def cos_avg_pool1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> torch.Tensor:
    if not input.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    
    cos_input = torch.cos(input).contiguous()
    B, C, iW = cos_input.shape
    stride = kernel_size if stride is None else stride

    numerator = iW + 2 * padding - kernel_size
    if numerator < 0:
        oW = 0
    else:
        if ceil_mode:
            oW = (numerator + stride - 1) // stride
        else:
            oW = numerator // stride
        oW += 1

    if oW <= 0:
        return torch.empty((B, C, 0), device=input.device)
    
    output = torch.empty((B, C, oW), device=input.device)
    BLOCK_SIZE = 256
    num_blocks = (oW + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_instances = B * C
    grid = (total_instances, num_blocks)
    count_include_pad_int = 1 if count_include_pad else 0

    cos_avg_pool1d_kernel[grid](
        input_ptr=cos_input,
        output_ptr=output,
        B=B,
        C=C,
        iW=iW,
        oW=oW,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad_int,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
