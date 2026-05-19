import torch
import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(x, mean, N, stride, BLOCK_SIZE: tl.constexpr):
    # Get the index of the current thread
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load elements from global memory
    x_block_ptr = x + offsets
    x_values = tl.load(x_block_ptr, mask=mask, other=0.0)

    # Compute exponential values
    x_exp = tl.exp(x_values)

    # Compute the sum of exponential values
    x_sum = tl.sum(x_exp, axis=0)

    # Write the sum to global memory
    if pid == 0:
        mean[0] = x_sum

    # Load the sum again (to avoid bank conflict)
    x_sum = tl.load(mean)

    # Compute the mean
    mean_value = x_sum / N
    tl.store(mean + stride, mean_value)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if dtype is None:
        dtype = input.dtype

    if out is None:
        out = torch.empty(input.shape, dtype=dtype, device=input.device)
    elif out.dtype != dtype:
        raise RuntimeError("exp_mean: expected dtype to be {} but got {}".format(dtype, out.dtype))

    if dim is None or len(dim) == input.ndim:
        input = input.flatten()
        out = out.flatten()
        N = input.numel()
        mean = torch.empty([1], dtype=dtype, device=input.device)
        grid = lambda meta: (1,)
    else:
        shape = list(input.shape)
        dim = [d % input.ndim for d in dim]
        input = dim_compress(input, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        out = out.view(shape)
        mean = torch.empty(shape, dtype=dtype, device=input.device)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    exp_mean_kernel[grid](input, mean, N, out.numel())
    if not keepdim:
        out = out.squeeze(dim)
    return out
