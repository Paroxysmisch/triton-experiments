import torch
import triton
import triton.language as tl

@triton.jit
def _silu_batch_norm_kernel(
    input_ptr, 
    output_ptr, 
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    n_elements,
    eps,
    c,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Map linear index -> channel index
    # Assumes input is flattened, original shape is [N, C, ...].
    channel = offsets % c

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mean_ptr + channel, mask=mask, other=0.0)
    v = tl.load(var_ptr + channel, mask=mask, other=0.0)

    w = 1.0
    b = 0.0
    if weight_ptr != 0:
        w = tl.load(weight_ptr + channel, mask=mask, other=1.0)
    if bias_ptr != 0:
        b = tl.load(bias_ptr + channel, mask=mask, other=0.0)

    # Batch Norm
    x_hat = (x - m) * tl.rsqrt(v + eps)
    bn_out = x_hat * w + b

    # SiLU activation: out = x * sigmoid(x)
    out = bn_out * (1.0 / (1.0 + tl.exp(-bn_out)))

    tl.store(output_ptr + offsets, out, mask=mask)


def silu_batch_norm(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Applies Batch Normalization over an input tensor across channels,
    followed by the SiLU activation function applied element-wise.
    """
    # Determine channel dimension (assume input.shape = [N, C, ...])
    # Flatten everything except keep track of total elements
    input_shape = input.shape
    if input.dim() < 2:
        raise ValueError("Expected input with at least 2 dimensions [N, C, ...].")

    # Compute current mean/var for training or use running mean/var for eval
    if training:
        # Compute mean, var across [N, H, W, ...], for each channel
        dim = [0] + list(range(2, input.dim()))
        curr_mean = input.mean(dim=dim)
        curr_var = input.var(dim=dim, unbiased=False)
        with torch.no_grad():
            running_mean[:] = (1 - momentum) * running_mean + momentum * curr_mean
            running_var[:] = (1 - momentum) * running_var + momentum * curr_var
        mean = curr_mean
        var = curr_var
    else:
        mean = running_mean
        var = running_var

    # Flatten input for kernel processing
    cdim = input_shape[1]
    x = input.contiguous().view(-1)
    out = torch.empty_like(x)

    # Handle None weight/bias by substituting pointer=0 in kernel
    weight_ptr = weight if weight is not None else 0
    bias_ptr = bias if bias is not None else 0

    grid = lambda meta: ((x.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _silu_batch_norm_kernel[grid](
        x, 
        out,
        mean, 
        var,
        weight_ptr, 
        bias_ptr, 
        x.numel(), 
        eps, 
        cdim,
        BLOCK_SIZE=1024
    )
    return out.view(input_shape)
