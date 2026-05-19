import triton
from triton import cdiv
import torch

@triton.jit
def fused_fractional_max_pool2d_with_relu_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    input_height,
    input_width,
    num_planes,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    pad_height,
    pad_width,
    return_indices,
    BLOCK_SIZE=32):
    
    # Get global indices
    pid = tl.program_id(axis=0)
    grid_size = cdiv(num_planes * input_height * input_width, BLOCK_SIZE)

    # Per-block indices
    num_threads_per_block = BLOCK_SIZE ** 2
    x = pid * num_threads_per_block + tl.arange(0, num_threads_per_block)
    n = x % (num_planes * input_height * input_width)
    c = x // (num_planes * input_height * input_width)

    if n >= num_planes * input_height * input_width:
        return

    in_idx = n
    out_c = c // (kernel_height * kernel_width)
    out_n = n // (kernel_height * kernel_width)

    in_h = out_c % input_height
    in_w = out_c // input_height

    max_val = -float('inf')
    max_idx = -1

    for kh in range(kernel_height):
        for kw in range(kernel_width):
            h = in_h * stride_height + kh - pad_height
            w = in_w * stride_width + kw - pad_width

            if 0 <= h < input_height and 0 <= w < input_width:
                idx = h * input_width + w
                val = tl.relu(input_ptr[in_idx * input_height * input_width + idx])
                if val > max_val:
                    max_val = val
                    max_idx = idx

    output_ptr[out_c * input_height * input_width + n] = max_val

    if return_indices:
        indices_ptr[out_c * input_height * input_width + n] = max_idx

# Triton C++ API wrapper
@triton.jit
def fused_fractional_max_pool2d_with_relu_wrapper(
    input_ptr,
    output_ptr,
    indices_ptr,
    input_height,
    input_width,
    num_planes,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    pad_height,
    pad_width,
    return_indices,
    BLOCK_SIZE=32):
    
    fused_fractional_max_pool2d_with_relu_kernel[input_numel=(num_planes * input_height * input_width), block=BLOCK_SIZE](input_ptr, output_ptr, indices_ptr, input_height, input_width, num_planes, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, return_indices)

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    input_height, input_width = input.shape[1:]
    num_planes = input.shape[0]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    kernel_height, kernel_width = kernel_size
    
    if output_size is None and output_ratio is None:
        raise ValueError("Either output_size or output_ratio must be specified")
    
    if output_size is not None:
        out_height, out_width = output_size
    elif output_ratio is not None:
        out_height = int(input_height * output_ratio[0])
        out_width = int(input_width * output_ratio[1])

    stride_height = kernel_height
    stride_width = kernel_width
    pad_height = 0
    pad_width = 0

    # Allocate memory for output and indices
    output = torch.empty((num_planes, out_height, out_width), dtype=torch.float32, device=input.device)
    indices = torch.empty((num_planes, out_height, out_width), dtype=torch.int32, device=input.device) if return_indices else None

    # Convert input to Triton format
    input_triton = input.contiguous().view(-1).to(device='cuda').numpy()
    output_triton = output.view(-1).to(device='cuda').numpy()
    indices_triton = indices.view(-1).to(device='cuda').numpy() if indices is not None else None

    # Launch Triton kernel
    fused_fractional_max_pool2d_with_relu_wrapper[
        grid=num_planes * out_height * out_width,
        block=32
    ](input_triton, output_triton, indices_triton, input_height, input_width, num_planes, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, return_indices)

    # Convert output back to PyTorch format
    output = torch.from_numpy(output_triton).reshape(num_planes, out_height, out_width).to(dtype=torch.float32, device=input.device)
    if indices is not None:
        indices = torch.from_numpy(indices_triton).reshape(num_planes, out_height, out_width).to(dtype=torch.int32, device=input.device)

    return output if not return_indices else (output, indices)
