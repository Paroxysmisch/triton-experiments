import torch
import triton
import triton.language as tl

# Triton kernel for computing the cosine of each element in the input tensor
@triton.jit
def compute_cosine(
    input_ptr, output_ptr, size: tl.constexpr
):
    """
    Computes the cosine of each element in the input tensor.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    size : tl.constexpr
        The number of elements in the input tensor.
    """
    pid = tl.program_id(0)
    block_size = 256
    block_start = pid * block_size
    offsets = tl.arange(0, block_size)
    idx = block_start + offsets

    mask = idx < size
    input_val = tl.load(input_ptr + idx, mask=mask)
    output_val = tl.cos(input_val)
    tl.store(output_ptr + idx, output_val, mask=mask)

# Triton kernel for 1D average pooling
@triton.jit
def avg_pool1d(
    input_ptr, output_ptr, input_size, output_size, kernel_size, stride, padding, count_include_pad: tl.constexpr
):
    """
    Applies 1D average pooling over the input tensor.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    input_size : int
        The size of the input tensor along the pooling dimension.
    output_size : int
        The size of the output tensor along the pooling dimension.
    kernel_size : int
        The size of the pooling window.
    stride : int
        The stride of the pooling window.
    padding : int
        The amount of zero-padding added to both sides of the input.
    count_include_pad : tl.constexpr
        If True, includes the zero-padding in the averaging calculation.
    """
    pid = tl.program_id(0)
    block_size = 256
    block_start = pid * block_size
    offsets = tl.arange(0, block_size)
    idx = block_start + offsets

    mask = idx < output_size
    output_val = tl.zeros((block_size,), dtype=tl.float32)

    for k in range(kernel_size):
        input_idx = idx * stride + k - padding
        input_mask = (input_idx >= 0) & (input_idx < input_size) & mask
        input_val = tl.load(input_ptr + input_idx, mask=input_mask, other=0.0)
        output_val += input_val

    if count_include_pad:
        output_val /= kernel_size
    else:
        valid_count = tl.sum(input_mask, axis=0)
        output_val /= valid_count

    tl.store(output_ptr + idx, output_val, mask=mask)

# Wrapper function for cos_avg_pool1d
def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    """
    Applies the cosine function element-wise to the input tensor, followed by 1D average pooling.

    Parameters:
    -----------
    input (Tensor): The input tensor of shape (minibatch, in_channels, iW).
    kernel_size (int): Size of the pooling window.
    stride (int, optional): Stride of the pooling window. Defaults to `kernel_size`.
    padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
    ceil_mode (bool, optional): If True, uses ceil instead of floor to compute the output shape. Default is False.
    count_include_pad (bool, optional): If True, includes the zero-padding in the averaging calculation. Default is True.

    Returns:
    --------
    torch.Tensor
        The output tensor after applying cosine and 1D average pooling.
    """
    if stride is None:
        stride = kernel_size

    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for triton ops."

    B, C, W = input.shape
    input = input.view(-1, W)  # Flatten the input to (B * C, W)

    # Compute the cosine of each element in the input tensor
    cosine_output = torch.empty_like(input)
    grid = (input.numel() // 256 + 1,)
    compute_cosine[grid](input, cosine_output, input.numel())

    # Compute the output size for 1D average pooling
    if ceil_mode:
        output_size = (W + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        output_size = (W + 2 * padding - kernel_size) // stride + 1

    # Apply 1D average pooling
    pooled_output = torch.empty((input.shape[0], output_size), device=device)
    grid = (input.shape[0] * output_size // 256 + 1,)
    avg_pool1d[grid](cosine_output, pooled_output, W, output_size, kernel_size, stride, padding, count_include_pad)

    # Reshape the output back to (B, C, output_size)
    pooled_output = pooled_output.view(B, C, output_size)

    return pooled_output
