import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def cosine_kernel(input_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    """
    Computes the cosine of each element in the input tensor.
    
    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    size : int
        Number of elements in the input tensor.
    BLOCK_SIZE : tl.constexpr
        Number of elements processed per block.
    """
    idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    mask = idx < size
    input_val = tl.load(input_ptr + idx, mask=mask)
    output_val = tl.cos(input_val)
    tl.store(output_ptr + idx, output_val, mask=mask)

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    """
    Applies the cosine function element-wise to the input tensor, followed by a 1D average pooling.
    
    Parameters:
    -----------
    input : torch.Tensor
        The input tensor of shape (minibatch, in_channels, iW).
    kernel_size : int
        Size of the pooling window.
    stride : int, optional
        Stride of the pooling window. Defaults to `kernel_size`.
    padding : int, optional
        Zero-padding added to both sides of the input. Default is 0.
    ceil_mode : bool, optional
        If True, uses ceil instead of floor to compute the output shape. Default is False.
    count_include_pad : bool, optional
        If True, includes the zero-padding in the averaging calculation. Default is True.
        
    Returns:
    --------
    torch.Tensor
        The result of applying the cosine function and then average pooling.
    """
    assert input.device.type == 'cuda', "Input tensor must be on a CUDA device for Triton ops."

    # Get input dimensions
    B, C, W = input.shape
    output = torch.empty_like(input)

    # Triton kernel launch configuration
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(B * C * W, meta['BLOCK_SIZE']),)

    # Compute cosine of input using Triton
    cosine_kernel[grid](input_ptr=input, output_ptr=output, size=B * C * W, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape the output for pooling
    output = output.view(B, C, W)

    # Perform 1D average pooling using PyTorch
    if stride is None:
        stride = kernel_size
    pooled_output = F.avg_pool1d(output, kernel_size, stride, padding, ceil_mode, count_include_pad)

    return pooled_output
