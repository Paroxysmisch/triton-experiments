import torch
import triton
import triton.language as tl

# Triton kernel for adaptive average pooling followed by sigmoid activation
@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_shape,  # Shape of the input tensor (N, C, H, W)
    output_shape,  # Shape of the output tensor (N, C, H_out, W_out)
    stride_h,  # Stride in the height dimension
    stride_w,  # Stride in the width dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    # Get the current block's position in the grid
    pid = tl.program_id(axis=0)
    batch, channel, height, width = input_shape
    _, _, height_out, width_out = output_shape

    # Compute the output coordinates for this block
    h_out = (pid % height_out) * BLOCK_SIZE
    w_out = (pid // height_out) * BLOCK_SIZE

    # Compute the input coordinates for the pooling region
    h_start = h_out * stride_h
    h_end = min(h_start + stride_h, height)
    w_start = w_out * stride_w
    w_end = min(w_start + stride_w, width)

    # Initialize the sum and count for the pooling region
    sum_val = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    count = (h_end - h_start) * (w_end - w_start)

    # Iterate over the pooling region
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_offset = (pid % batch) * channel * height * width + \
                           (pid // batch) * height * width + \
                           h * width + w
            sum_val += tl.load(input_ptr + input_offset)

    # Compute the average value for the pooling region
    avg_val = sum_val / count

    # Apply the sigmoid activation
    sigmoid_val = 1 / (1 + tl.exp(-avg_val))

    # Store the result in the output tensor
    output_offset = (pid % batch) * channel * height_out * width_out + \
                    (pid // batch) * height_out * width_out + \
                    h_out * width_out + w_out
    tl.store(output_ptr + output_offset, sigmoid_val)

# Python wrapper function
def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    input = input.to(device='cuda')

    # Determine the output size
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    height_out, width_out = output_size

    # Get the input shape
    batch, channel, height, width = input.shape

    # Compute the stride for the pooling
    stride_h = height // height_out
    stride_w = width // width_out

    # Allocate the output tensor
    output = torch.empty((batch, channel, height_out, width_out), device='cuda')

    # Launch the Triton kernel
    grid = (batch * channel * height_out * width_out, )
    sigmoid_adaptive_avg_pool2d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        input_shape=(batch, channel, height, width),
        output_shape=(batch, channel, height_out, width_out),
        stride_h=stride_h,
        stride_w=stride_w,
        BLOCK_SIZE=1,
    )

    return output

# Example usage
input_tensor = torch.randn(2, 3, 16, 16, device='cuda')
output_tensor = sigmoid_adaptive_avg_pool2d(input_tensor, (4, 4))
print(output_tensor)
