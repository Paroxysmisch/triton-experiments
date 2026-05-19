import torch
import triton
import triton.language as tl

@triton.jit
def _exp_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _sum_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A simple parallel sum kernel for demonstration.
    Each program_id(0) processes one block of data. We accumulate partial sums
    into a single element and then store it at the location indexed by program_id(0).
    A final pass in Python can then sum across these partial sums if multiple blocks are used.
    """
    pid = tl.program_id(0)
    # Start and end indices for this block
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    # Perform an in-block reduction
    # Step 1: local store
    reduction = tl.sum(x, axis=0)
    # Store the partial sum in out_ptr (one result per block)
    tl.store(out_ptr + pid, reduction)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    # Ensure input is on GPU
    x = input if input.is_cuda else input.cuda()
    # Compute exponent via Triton kernel
    exp_res = torch.empty_like(x, dtype=(dtype if dtype else x.dtype))

    BLOCK_SIZE = 1024
    n_elements = x.numel()
    # Launch exp kernel
    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _exp_kernel[grid](x, exp_res, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Functionality for summation and mean
    if dim is None:
        # Flatten for total mean
        flat_exp = exp_res.view(-1)
        n = flat_exp.shape[0]
        # Sum the entire array with multiple blocks if large
        # Compute blocks needed
        blocks_needed = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        partial_sums = torch.empty(blocks_needed, dtype=exp_res.dtype, device=exp_res.device)
        _sum_kernel[blocks_needed](
            flat_exp, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE
        )
        total_sum = partial_sums.sum()
        mean_val = total_sum / n
        out_tensor = mean_val if out is None else out.copy_(mean_val)
    else:
        # Use PyTorch's dimension-based mean on the exponent result for simplicity
        # but this could be replaced with a dimension-wise Triton reduction kernel.
        exp_dim_mean = exp_res.mean(dim=dim, keepdim=keepdim)
        if dtype is not None:
            exp_dim_mean = exp_dim_mean.to(dtype)
        out_tensor = exp_dim_mean if out is None else out.copy_(exp_dim_mean)

    return out_tensor if out is not None else out_tensor.clone()
