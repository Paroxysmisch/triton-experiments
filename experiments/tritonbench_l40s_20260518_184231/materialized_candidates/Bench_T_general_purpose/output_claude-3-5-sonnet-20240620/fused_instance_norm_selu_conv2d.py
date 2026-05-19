import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_conv2d_selu_instancenorm_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    # Matrix dimensions
    batch, in_channels, out_channels, in_height, in_width,
    kernel_height, kernel_width,
    # Parameters
    stride, padding, dilation, groups,
    # Instance norm parameters
    eps,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Calculate position in output matrix
    pid = tl.program_id(0)
    
    # Load the input block
    input_block_ptr = tl.make_block_ptr(
        input_ptr,
        shape=(batch, in_channels, in_height, in_width),
        strides=(in_channels * in_height * in_width, in_height * in_width, in_width, 1),
        offsets=(0, 0, 0, 0),
        block_shape=(1, BLOCK_SIZE_M, BLOCK_SIZE_N, 1),
        order=(1, 2, 3, 0)
    )
    
    # Load input block
    x = tl.load(input_block_ptr)
    
    # Convolution computation
    # ... (convolution implementation)
    
    # SELU activation
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    mask = x <= 0
    x = tl.where(mask, alpha * (tl.exp(x) - 1.0), x)
    x = scale * x
    
    # Instance Normalization
    mean = tl.sum(x, axis=1, keepdims=True) / (in_height * in_width)
    var = tl.sum((x - mean) ** 2, axis=1, keepdims=True) / (in_height * in_width)
    x = (x - mean) / tl.sqrt(var + eps)
    
    # Store result
    tl.store(output_ptr + pid * BLOCK_SIZE_M, x)

def fused_instance_norm_selu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    num_features: int = None,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = False,
    track_running_stats: bool = False
) -> torch.Tensor:
    # Input validation
    assert input.dim() == 4, "Input must be a 4D tensor"
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=input.device,
        dtype=input.dtype
    )
    
    # Handle bias
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
    
    # Grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 8
    
    grid = lambda meta: (
        triton.cdiv(out_channels, BLOCK_SIZE_M) *
        triton.cdiv(out_height * out_width, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    fused_conv2d_selu_instancenorm_kernel[grid](
        input.contiguous(), weight.contiguous(), output,
        bias.contiguous() if bias is not None else None,
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_height, kernel_width,
        stride, padding, dilation, groups, eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output
