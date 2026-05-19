import torch
import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(input_ptr, output_ptr, in_h, in_w, out_h, out_w, stride_h, stride_w, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    h_start = pid_h * stride_h
    w_start = pid_w * stride_w

    h_end = tl.min(h_start + stride_h, in_h)
    w_end = tl.min(w_start + stride_w, in_w)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    count = 0

    # Loop over the pooling window
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_offset = h * in_w + w
            acc += tl.load(input_ptr + input_offset)
            count += 1

    # Compute average
    acc /= count

    # Apply sigmoid function
    acc = 1 / (1 + tl.exp(-acc))

    # Store the result
    output_offset = pid_h * out_w + pid_w
    tl.store(output_ptr + output_offset, acc)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    in_h, in_w = input.shape[-2], input.shape[-1]
    out_h, out_w = output_size

    stride_h = in_h // out_h
    stride_w = in_w // out_w

    output = torch.empty((out_h, out_w), device=input.device, dtype=input.dtype)

    grid = (out_h, out_w)

    triton_kernel = adaptive_avg_pool2d_kernel[grid](
        input, output, in_h, in_w, out_h, out_w, stride_h, stride_w,
        BLOCK_H=1, BLOCK_W=1  # You can adjust these block sizes based on your use case
    )

    return output

# Example usage
input_tensor = torch.randn(32, 32, device='cuda')
output_tensor = sigmoid_adaptive_avg_pool2d(input_tensor, (8, 8))
print(output_tensor)
