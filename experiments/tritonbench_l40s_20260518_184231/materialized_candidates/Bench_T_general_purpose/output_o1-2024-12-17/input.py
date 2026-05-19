import math
import torch
import triton
import triton.language as tl

@triton.jit
def _mean_kernel(
    input_ptr,  # pointer to input data
    output_ptr, # pointer to output data
    N,          # number of elements to reduce
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Perform block-level sum
    acc = tl.sum(x, axis=0)
    # Reduce sum across the warp
    acc = tl.sum(tl.broadcast_to(acc, [BLOCK_SIZE]), axis=0)
    # Compute mean for this block
    count = tl.sum(mask, axis=0)
    mean_val = acc / tl.max(count, 1)
    # Store result
    if pid == 0:
        tl.store(output_ptr, mean_val)

def trion_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    # Handle optional dtype
    if dtype is not None:
        input = input.to(dtype)

    if dim is None:
        # Reduce over all dims
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)

    # Repeatedly reduce one dimension at a time
    result = input
    for d in sorted(dim, reverse=True):
        # Move dim d to the last dimension
        perm = list(range(result.ndim))
        perm[d], perm[-1] = perm[-1], perm[d]
        result = result.permute(perm).contiguous()
        shape = result.shape
        # Flatten all but the last dimension
        leading = math.prod(shape[:-1])
        trailing = shape[-1]

        # Allocate temporary storage for one reduction
        tmp_out = torch.empty((leading,), device=result.device, dtype=result.dtype)
        # Launch Triton kernel
        grid = lambda meta: ( (leading + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
        _mean_kernel[grid](result.data_ptr(),
                           tmp_out.data_ptr(),
                           trailing,
                           BLOCK_SIZE=1024)
        # Reshape back
        tmp_out = tmp_out.reshape(*shape[:-1])
        # Move the last dimension back to dimension d
        inv_perm = list(range(len(perm)))
        inv_perm[-1], inv_perm[d] = inv_perm[d], inv_perm[-1]
        tmp_out = tmp_out.permute(inv_perm)
        # Insert dimension if keepdim
        if keepdim:
            tmp_out = tmp_out.unsqueeze(d)
        result = tmp_out

    # Prepare final output
    if out is not None:
        out.resize_(result.shape)
        out.copy_(result)
        return out
    return result
