import triton
import triton.language as tl
import torch

@triton.jit
def _add_kernel(
    input_ptr, other_ptr, out_ptr,
    size, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    inp = tl.load(input_ptr + offsets, mask=mask, other=0)
    oth = tl.load(other_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, inp + alpha * oth, mask=mask)

@triton.jit
def _reduce_mean_kernel(
    x_ptr, out_ptr,
    size,
    BLOCK_SIZE: tl.constexpr
):
    # Sum in chunks of BLOCK_SIZE
    sum_val = tl.zeros([], dtype=tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)
    for start in range(0, size, BLOCK_SIZE):
        mask = offsets < (size - start)
        val = tl.load(x_ptr + (start + offsets), mask=mask, other=0.)
        sum_val += tl.sum(val, where=mask)
    mean_val = sum_val / size
    tl.store(out_ptr + 0, mean_val)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    """
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes
    the mean value along the specified dimension. If no dimension is specified,
    computes the mean over all elements. Supports broadcasting, type promotion,
    and works with integer, float, and complex inputs.
    """
    if dtype is not None:
        input = input.to(dtype)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other = other.to(input.device, dtype=input.dtype)

    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)
    input_exp = input.expand(broadcast_shape)
    other_exp = other.expand(broadcast_shape)

    if out is None:
        out = torch.empty_like(input_exp)

    # Flatten for kernel
    input_flat = input_exp.reshape(-1)
    other_flat = other_exp.reshape(-1)
    out_flat = out.reshape(-1)
    size = input_flat.numel()

    BLOCK_SIZE = 1024
    grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _add_kernel[grid](
        input_flat, other_flat, out_flat,
        size, alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # If dim is None, compute mean over all elements
    if dim is None:
        temp_out = torch.empty((1,), dtype=out.dtype, device=out.device)
        _reduce_mean_kernel[(1,)](
            out_flat, temp_out,
            size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        mean_val = temp_out[0]
        if keepdim:
            # Return a tensor of the same number of dimensions but 1 in each dim
            return mean_val.reshape([1] * len(broadcast_shape))
        else:
            return mean_val

    # If dim is specified, shortcut via PyTorch's mean on the data buffered in out
    return out.mean(dim=dim, keepdim=keepdim)
