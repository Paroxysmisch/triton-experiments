import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_height, input_width, input_channels,
    output_height, output_width, output_channels,
    kernel_height, kernel_width,
    stride_height, stride_width,
    padding_height, padding_width,
    groups, 
    BLOCK_SIZE_BATCH: tl.constexpr, 
    BLOCK_SIZE_INPUT: tl.constexpr, 
    BLOCK_SIZE_OUTPUT: tl.constexpr,
    USE_FP16: tl.constexpr, 
    USE_TF32: tl.constexpr
):
    batch_idx = tl.program_id(0)
    output_channel_idx = tl.program_id(1)
    output_x = tl.program_id(2)
    output_y = tl.program_id(3)
    
    batch_start = batch_idx * BLOCK_SIZE_BATCH
    output_channel_start = output_channel_idx * BLOCK_SIZE_OUTPUT
    output_x_start = output_x * BLOCK_SIZE_INPUT
    output_y_start = output_y * BLOCK_SIZE_INPUT
    
    # Compute input and output positions
    input_x_start = output_x_start * stride_width - padding_width
    input_y_start = output_y_start * stride_height - padding_height
    
    for b in range(BLOCK_SIZE_BATCH):
        for oc in range(BLOCK_SIZE_OUTPUT):
            if batch_start + b < input_ptr.shape[0] and output_channel_start + oc < output_channels:
                acc = 0.0
                for ic in range(input_channels // groups):
                    for ky in range(kernel_height):
                        for kx in range(kernel_width):
                            input_x = input_x_start + kx
                            input_y = input_y_start + ky
                            if 0 <= input_x < input_width and 0 <= input_y < input_height:
                                input_idx = ((batch_start + b) * input_channels + ic) * input_height * input_width + input_y * input_width + input_x
                                weight_idx = ((output_channel_start + oc) * (input_channels // groups) + ic) * kernel_height * kernel_width + ky * kernel_width + kx
                                acc += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)
                output_idx = ((batch_start + b) * output_channels + output_channel_start + oc) * output_height * output_width + output_y_start * output_width + output_x_start
                tl.store(output_ptr + output_idx, acc)

def conv2d_forward(
    input: torch.Tensor, weight: torch.Tensor, 
    kernel_size: Tuple[int, int], stride: Tuple[int, int], 
    padding: Tuple[int, int], groups: int, 
    use_fp16: bool = False, use_tf32: bool = False
):
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, _, kernel_height, kernel_width = weight.shape
    
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    
    output = torch.empty((batch_size, output_channels, output_height, output_width), device=input.device, dtype=input.dtype)
    
    BLOCK_SIZE_BATCH = 1
    BLOCK_SIZE_INPUT = 1
    BLOCK_SIZE_OUTPUT = 1
    
    grid = (batch_size // BLOCK_SIZE_BATCH, output_channels // BLOCK_SIZE_OUTPUT, output_width // BLOCK_SIZE_INPUT, output_height // BLOCK_SIZE_INPUT)
    
    conv2d_forward_kernel[grid](
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        input_height, input_width, input_channels,
        output_height, output_width, output_channels,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width,
        groups,
        BLOCK_SIZE_BATCH, BLOCK_SIZE_INPUT, BLOCK_SIZE_OUTPUT,
        use_fp16, use_tf32
    )
    
    return output
