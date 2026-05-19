import triton
import triton.language as tl
import torch

@triton.jit
def _batch_norm_fwd_kernel(
    x_ptr,  # [batch_size * channels * ...]
    mean_ptr,
    var_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    total_elems,
    num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Determine channel index from flat index
    # Assumes data shape [N, C, *spatial] flattened
    # channel_idx = (offset // per_channel_size) % C
    # Here, we compute it directly from 'offsets'
    channel_idx = offsets % num_channels

    # Load mean, var for this channel
    mean_val = tl.load(mean_ptr + channel_idx)
    var_val = tl.load(var_ptr + channel_idx)

    # Normalize
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    x_norm = (x - mean_val) * inv_std

    # Scale/Shift
    if w_ptr != 0:
        w_val = tl.load(w_ptr + channel_idx)
        x_norm *= w_val
    if b_ptr != 0:
        b_val = tl.load(b_ptr + channel_idx)
        x_norm += b_val

    tl.store(y_ptr + offsets, x_norm, mask=mask)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    # Ensure input is a Torch tensor on a CUDA device
    x = input.contiguous()
    device = x.device
    dtype = x.dtype

    # Flatten everything except for channels
    # Assume shape = [N, C, *spatial]
    shape = x.shape
    if len(shape) < 2:
        raise ValueError("Input must have at least 2 dimensions [N, C].")
    N, C = shape[0], shape[1]
    numel = x.numel()
    x_flat = x.view(-1)

    # Convert running stats to contiguous tensors as well
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()

    # If training, compute mean & var over the input
    if training:
        x_reshaped = x.permute(1, 0, *range(2, len(shape))).contiguous().view(C, -1)
        batch_mean = x_reshaped.mean(dim=1)
        batch_var = x_reshaped.var(dim=1, unbiased=False)

        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * batch_mean)
            running_var.mul_(1 - momentum).add_(momentum * batch_var)
    else:
        batch_mean = running_mean
        batch_var = running_var

    # Prepare output
    y = torch.empty_like(x)
    y_flat = y.view(-1)

    # Convert weight, bias to pointers if they exist, else pass 0
    w_ptr = 0
    b_ptr = 0
    if weight is not None:
        weight = weight.contiguous()
        w_ptr = weight.data_ptr()
    if bias is not None:
        bias = bias.contiguous()
        b_ptr = bias.data_ptr()

    # Launch kernel
    grid = lambda meta: ( (numel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _batch_norm_fwd_kernel[grid](
        x_flat.data_ptr(),
        batch_mean.data_ptr(),
        batch_var.data_ptr(),
        w_ptr,
        b_ptr,
        y_flat.data_ptr(),
        numel,
        C,
        eps,
        BLOCK_SIZE=1024
    )
    return y.reshape(shape)
