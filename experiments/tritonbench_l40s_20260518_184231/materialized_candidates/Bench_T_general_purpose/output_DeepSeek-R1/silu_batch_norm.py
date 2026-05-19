import torch
import triton
import triton.language as tl

@triton.jit
def silu_batch_norm_kernel(
    input_ptr, output_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, H, W, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    hw = H * W
    linear_idx_div = offsets // hw
    c_indices = linear_idx_div % C

    mean = tl.load(mean_ptr + c_indices, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_indices, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c_indices, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_indices, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (input_vals - mean) * inv_std
    scaled = normalized * weight
    shifted = scaled + bias

    sig = 1.0 / (1.0 + tl.exp(-shifted))
    silu = shifted * sig

    tl.store(output_ptr + offsets, silu, mask=mask)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    assert input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"
    if training:
        mean = input.mean(dim=(0, 2, 3))
        var = input.var(dim=(0, 2, 3), unbiased=False)
        running_mean.copy_((1 - momentum) * running_mean + momentum * mean.detach())
        running_var.copy_((1 - momentum) * running_var + momentum * var.detach())
    else:
        mean = running_mean
        var = running_var

    weight = torch.ones_like(mean) if weight is None else weight
    bias = torch.zeros_like(mean) if bias is None else bias

    N, C, H, W = input.shape
    input_flat = input.contiguous().view(-1)
    output = torch.empty_like(input)
    output_flat = output.view(-1)
    n_elements = input_flat.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_batch_norm_kernel[grid](
        input_flat, output_flat, mean, var, weight, bias,
        n_elements, C, H, W, eps,
        BLOCK_SIZE=1024
    )
    return output
