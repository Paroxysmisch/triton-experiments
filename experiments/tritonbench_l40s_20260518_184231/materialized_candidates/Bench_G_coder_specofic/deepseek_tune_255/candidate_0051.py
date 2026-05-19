import triton
import triton.language as tl
import torch

# Triton kernel to compute max in a large 1D tensor
@triton.jit
def max_kernel_1(
    inp,
    mid,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Create a mask to prevent out-of-bound memory access
    mask = offset < inp.shape[0]
    # Load values with the mask
    values = tl.load(inp + offset, mask=mask)
    # Compute the maximum value
    max_value = tl.max(values)
    # Store the maximum value in mid
    tl.store(mid + pid, max_value, mask=True)

# Triton kernel to consolidate max computations
@triton.jit
def max_kernel_2(mid, out, MID_SIZE: tl.constexpr):
    # Load all mid values
    values = tl.load(mid + tl.arange(0, MID_SIZE))
    # Compute the maximum value
    max_value = tl.max(values)
    # Store the maximum value in out
    tl.store(out, max_value)

# Wrapper function to execute max_kernel_1 and max_kernel_2
def max(inp):
    BLOCK_SIZE = triton.next_power_of_2(inp.shape[0])
    mid = torch.empty((BLOCK_SIZE,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=torch.float32, device=inp.device)
    # Ensure optimal number of blocks
    grid = lambda meta: (triton.cdiv(inp.shape[0], meta["BLOCK_SIZE"]),)
    # Max kernel 1
    max_kernel_1[grid](inp, mid, BLOCK_SIZE=BLOCK_SIZE)
    # Max kernel 2
    max_kernel_2[grid](mid, out, MID_SIZE=BLOCK_SIZE)
    return out

# Triton kernel for multi-dimensional max computation
@triton.jit
def max_kernel(
    inp,
    out,
    dim,
    M,
    K,
    pre_dim_shape,
    post_dim_shape,
    pre_dim_stride,
    post_dim_stride,
    pid_m: tl.constexpr,
    pid_k: tl.constexpr,
):
    offset_m = pid_m * M
    offset_k = pid_k * K
    arange_m = offset_m + tl.arange(0, M)
    arange_k = offset_k + tl.arange(0, K)
    # Load data with masking
    mask = (arange_m < inp.shape[dim]) & (arange_k < post_dim_shape)
    inp_view = tl.reshape(inp, pre_dim_shape + post_dim_shape)
    inp_ptrs = inp_view + (arange_m * pre_dim_stride + arange_k * post_dim_stride)
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    # Compute max and argmax
    max_val = tl.max(inp_val)
    # Store output
    out_view = tl.reshape(out, pre_dim_shape + (1,))
    out_ptrs = out_view + (arange_m * pre_dim_stride)
    out_mask = arange_m < inp.shape[dim]
    tl.store(out_ptrs, max_val, mask=out_mask)

# Wrapper function to execute max_kernel
def max_dim(inp, dim, keepdim=False):
    # Check validity of dim
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError("Dimension out of range (expected to be in range of [{}, {}], but got {})".format(-inp.ndim, inp.ndim - 1, dim))
    dim = dim % inp.ndim
    # Get pre and post dim shapes
    pre_dim_shape = list(inp.shape[:dim])
    post_dim_shape = list(inp.shape[dim + 1 :])
    dim_shape = list(inp.shape[dim : dim + 1])
    pre_dim_stride = list(inp.stride()[:dim])
    post_dim_stride = list(inp.stride()[dim + 1 :])
    inp_reshaped = inp.reshape(pre_dim_shape + dim_shape + post_dim_shape)
    # Compute grid
    M = 1
    for m in pre_dim_shape:
        M *= m
    K = dim_shape[0]
    post_dim_K = triton.cdiv(K, 1)
    grid = (M, post_dim_K)
    # Create output
    keepdim = keepdim or (dim_shape[0] > 1)
    if keepdim:
        out = torch.empty_like(inp)
    else:
        out = torch.empty([M] + list(post_dim_shape), dtype=inp.dtype, device=inp.device)
    # Max kernel
    max_kernel[grid](
        inp_reshaped,
        out,
        dim,
        M,
        K,
        pre_dim_shape,
        post_dim_shape,
        pre_dim_stride,
        post_dim_stride,
    )
    return out
