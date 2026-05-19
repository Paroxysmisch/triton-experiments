import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input,
    output,
    kernel_size,
    stride,
    padding,
    n,
    C,
    W,
    ceil_mode: tl.constexpr,
    count_include_pad: tl.constexpr,
):
    # Compute the index of the current output element
    pid = tl.program_id(0)
    oi = pid * stride
    oW = oi % W
    oC = (oi // W) % C
    # Compute the range of indices for the kernel
    kW = min(max(0, oW - kernel_size // 2), W - 1)
    kW = min(kernel_size, W - kW)
    # Compute the index of the input element corresponding to the current output element
    iW = oW - kernel_size // 2
    # Compute the sum of the cosine of the input elements within the kernel range
    cos_sum = 0.0
    cos_weight_sum = 0.0
    for j in range(0, kW):
        i = iW + j
        # Load the input element
        x = tl.load(input + n * C * W + oC * W + i)
        # Compute the cosine of the input element
        cos_x = tl.cos(x)
        # If count_include_pad is True, include the input element in the sum; otherwise, only include elements within the kernel
        if count_include_pad or (i >= padding and i < W - padding + kW - 1):
            cos_sum += cos_x
            cos_weight_sum += 1.0
    # Compute the average of the cosine of the input elements within the kernel range
    cos_avg = cos_sum / cos_weight_sum
    # Store the result in the output tensor
    tl.store(output + n * C * W + oC * W + oW, cos_avg)

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Handle default value for stride
    if stride is None:
        stride = kernel_size
    # Ensure input tensor has the correct shape
    assert input.dim() == 3
    # Extract tensor dimensions
    n, C, W = input.shape
    # Create output tensor
    output = torch.empty_like(input)
    # Define grid of kernel instances
    grid = lambda META: (triton.cdiv(W, META["stride"]),)
    # Dispatch Triton kernel
    cos_avg_pool1d_kernel[grid](
        input,
        output,
        kernel_size,
        stride,
        padding,
        n,
        C,
        W,
        count_include_pad=count_include_pad,
    )
    return output
