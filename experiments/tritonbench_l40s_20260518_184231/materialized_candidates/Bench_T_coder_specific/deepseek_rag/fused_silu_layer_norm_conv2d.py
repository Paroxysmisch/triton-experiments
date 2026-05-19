, conv_weight, conv_bias, conv_stride=1, conv_padding=1)
    >>> print(output.shape)
    torch.Size([4, 8, 32, 32])
other: Convolution operation parameters include stride, padding, dilation, and groups. Layer Normalization uses an epsilon value. Default values are provided for optional parameters.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to implement an RMS normalization kernel function `rms_norm_kernel` with 8 parameters: input, weight, output, input_row_stride, n_cols, eps, and two constexpr parameters N_COLS, BLOCK_N. The kernel normalizes input tensors using RMS normalization. The `trmsnorm` function calls the kernel and includes parameters hidden_states, weight, eps; it sets up the execution environment and manages the data preparation and launching of the kernel on specified grid sizes. import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride,
    n_cols,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < n_cols)
    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < n_cols)

@torch.inference_mode()
def trmsnorm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    """
    Applies the Triton RMSNorm operation to the given hidden states.

    Args:
        hidden_states (Tensor): The input hidden states.
        weight (Tensor): The weight tensor.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-6.

    Returns:
        Tensor: The output tensor after applying the RMSNorm operation.
    """

    def _kernel_meta():
        device = hidden_states.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)
    kernel_meta = _kernel_meta()
    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_stride,
        feat_size,
        eps,
        feat_size,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )
</s>
<|user|>
You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Applies 2D Convolution, followed by Layer Normalization and SiLU activation to the input tensor `x`. Sequentially performs convolution on `x`, then applies layer normalization on the convolution output, followed by SiLU activation applied element-wise.
Wrapper Entry Information: fused_silu_layer_norm_conv2d(x: torch.Tensor, weight: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor = None, conv_stride: int = 1, conv_padding: int = 0, conv_dilation: int = 1, conv_groups: int = 1, ln_eps: float = 1e-5) -> torch.Tensor

Args:
    x (Tensor): Input tensor for convolution, normalization, and activation.
    weight (Tensor): Learnable weight of size matching normalized output dimensions for LayerNorm.
    conv_weight (Tensor): Convolution kernel tensor of appropriate dimensions.
    conv_bias (Tensor, optional): Convolution bias tensor. Default: ``None``.
    conv_stride (int, optional): Stride of convolution. Default: 1.
    conv_padding (int, optional): Padding added to both sides of input. Default: 0.
    conv_dilation (int, optional): Dilation of convolution kernel. Default: 1.
    conv_groups (int, optional): Number of groups for convolution. Default: 1.
    ln_eps (float, optional): Epsilon value for Layer Normalization. Default: 1e-5.

Example:
    >>> x = torch.randn(4, 3, 32, 32)
    >>> conv_weight = torch.randn(8, 3, 3, 3)
    >>> conv_bias = torch.zeros(8)
    >>> weight = torch.ones(8)
    >>> output = fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias, conv_stride=1, conv_padding=1)
    >>> print(output.shape)
    torch.Size([4, 8, 32, 32])
other: Convolution operation parameters include stride, padding, dilation, and groups. Layer Normalization uses an epsilon value. Default values are provided for optional parameters.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to implement an RMS normalization kernel function `rms_norm_kernel` with 8 parameters: input, weight, output, input_row_stride, n_cols, eps, and two constexpr parameters N_COLS, BLOCK_N. The kernel normalizes input tensors using RMS normalization. The `trmsnorm` function calls the kernel and includes parameters hidden_states, weight, eps; it sets up the execution environment and manages the data preparation and launching of the kernel on specified grid sizes. import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride,
    n_cols,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < n_cols)
    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = x
