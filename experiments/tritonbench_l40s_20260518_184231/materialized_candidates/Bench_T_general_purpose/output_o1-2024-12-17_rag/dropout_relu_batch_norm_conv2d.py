import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Helper configs for Triton kernels (example block sizes).
# You can customize these based on your hardware or requirements.
def element_wise_kernel_configs():
    return [
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ]


@triton.jit
def apply_dropout(input_ptr, drop_p, seed, offset):
    random_val = tl.rand(seed, offset)
    return tl.where(random_val < drop_p, 0, input_ptr / (1 - drop_p))


@triton.jit
def apply_dropout_grad(output_grad_ptr, drop_p, seed, offset):
    random_val = tl.rand(seed, offset)
    return tl.where(random_val < drop_p, 0, output_grad_ptr / (1 - drop_p))


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_forward_kernel(
    input_pointer, output_pointer, size,
    drop_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    input_vals = tl.load(input_pointer + offsets, mask=mask)
    output_vals = apply_dropout(input_vals, drop_p, seed, offsets)
    tl.store(output_pointer + offsets, output_vals, mask=mask)


def dropout_forward(input_tensor: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    if (not training) or p <= 0.0:
        return input_tensor

    original_shape = input_tensor.shape
    flat_input = input_tensor.reshape(-1)
    flat_output = torch.empty_like(flat_input)

    size = flat_input.numel()
    seed = torch.randint(0, 2147483647, (1,)).item()
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (size + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

    dropout_forward_kernel[grid](flat_input, flat_output, size, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    out = flat_output.view(original_shape)
    return out


def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    p=0.5,
    training=True,
    inplace=False
) -> torch.Tensor:
    # 1) Convolution
    conv_out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # 2) Batch Normalization (using default running stats behavior here; can be extended as needed)
    # For an actual training scenario, you'd typically pass running_mean, running_var, etc.
    # Here, we'll use a simple approach with learnable parameters = None for demonstration.
    # If you want trainable batchnorm, you can create a module or pass in additional BN parameters.
    batch_out = F.batch_norm(
        conv_out,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=training,
        momentum=0.1,
        eps=1e-5
    )

    # 3) ReLU
    relu_out = F.relu(batch_out, inplace=inplace)

    # 4) Dropout
    out = dropout_forward(relu_out, p, training)

    return out
