@triton.jit
def _fused_hardsigmoid_batch_norm(
    x_ptr, mean_ptr, var_ptr, out_ptr, out_n, out_c, out_h, out_w,
    weight_ptr, bias_ptr, eps, scale_ptr, bias_ptr2,
    stride_in_bytes,
    _mma_layout=triton.jit.RegisterPrefetcher(
        triton.jit.Mma.col_major,
        triton.jit.Mma.alignment_4,
        triton.jit.Mma.access_policy_1,
        triton.jit.Mma.access_policy_1,
        triton.jit.Mma.access_policy_1,
    )
):
    BLOCK_SIZE = 256
    GROUP_SIZE = 8
    OUT_Y, OUT_X = triton.profiler.get_num_blocks('x', x.shape[0])
    OUT_Z = triton.profiler.get_num_blocks('x', x.shape[1])
    grid = launch_grid(OUT_Y, OUT_X, OUT_Z)

    # Load parameters
    eps = triton.jit.scalars.rsqrt(eps)
    weight = triton.jit.load(weight_ptr, stride_in_bytes)
    bias = triton.jit.load(bias_ptr, stride_in_bytes)

    # Perform batch normalization
    x_mean = triton.jit.load(mean_ptr, stride_in_bytes)
    x_var = triton.jit.load(var_ptr, stride_in_bytes)
    x = (x - x_mean) * triton.jit.scalars.rsqrt(x_var + eps)

    # Apply weight and bias
    x = x * weight + bias

    # Apply Hardsigmoid activation
    x = triton.jit.ops.hard_sigmoid(x)

    # Store output
    triton.jit.store(out_ptr, x, stride_in_bytes)

def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Check if inputs are on the correct device
    if not x.is_cuda or not running_mean.is_cuda or not running_var.is_cuda:
        raise ValueError("Inputs must be on the GPU")

    # Check if inputs are of the correct type
    if not isinstance(x, torch.Tensor) or not isinstance(running_mean, torch.Tensor) or not isinstance(running_var, torch.Tensor):
        raise ValueError("Inputs must be torch Tensors")

    # Check if inputs are of the correct shape
    if x.shape != running_mean.shape or x.shape != running_var.shape:
        raise ValueError("Inputs must have the same shape")

    # Check if weight and bias are of the correct shape
    if weight is not None and bias is not None and (weight.shape[0] != x.shape[1] or bias.shape[0] != x.shape[1]):
        raise ValueError("Weight and bias must have shape (C,)")

    # Check if inplace is a boolean
    if not isinstance(inplace, bool):
        raise ValueError("Inplace must be a boolean")

    # Perform batch normalization and activation
    _fused_hardsigmoid_batch_norm[grid](
        x.data_ptr(), running_mean.data_ptr(), running_var.data_ptr(),
        x.data_ptr() if inplace else torch.empty_like(x).data_ptr(),
        x.shape[0], x.shape[1], x.shape[2], x.shape[3],
        weight.data_ptr() if weight is not None else None,
        bias.data_ptr() if bias is not None else None,
        eps, None, None,
        x.element_size()
    )

    return x
