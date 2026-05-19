. the input tensors.
        ctx.save_for_backward(x, weight)
        # We know from the PyTorch API that x has shape
        # (N, H) and weight has shape (H,), so we can
        # infer that the output will have shape (N,).
        output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        # The kernel only works for BLOCK_SIZE >= H, so we
        # set a block size equal to the highest power of two
        # smaller than H.
        BLOCK_SIZE = 2**(x.shape[1].bit_length() - 1)
        # We launch the kernel with N instances.
        grid = (x.shape[0], )
        rms_norm_fwd[grid](x, weight, output, x.stride(0), x.shape[1], BLOCK_SIZE)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # The backward function receives the gradient of the loss
        # with respect to the output tensor, and we need to compute
        # the gradients with respect to the input tensors. The
        # context object contains the input tensors we need to access.
        x, weight = ctx.saved_tensors
        # The kernel only works for BLOCK_SIZE >= H, so we
        # set a block size equal to the highest power of two
        # smaller than H.
        BLOCK_SIZE = 2**(x.shape[1].bit_length() - 1)
        # We launch N instances of the kernel, where N is the
        # first dimension of the input tensors.
        grid = (x.shape[0], )
        # The output tensors are already allocated for us.
        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(weight)
        rms_norm_backward[grid](grad_output,
                                grad_x,
                                partial_grad_weight,
                                x,
                                weight,
                                x.stride(0),
                                x.shape[1],
                                1e-6,
                                BLOCK_SIZE)
        # We return the gradients with respect to both input tensors,
        # and None for the optional fourth input tensor.
        return grad_x, partial_grad_weight

def weighted_sum_triton(x, weight):
    return WeightedSumFunc_Triton.apply(x, weight)

@triton.jit
def rms_norm_fwd(x_ptr: tl.pointer_type,
                 weight_ptr: tl.pointer_type,
                 output_ptr: tl.pointer_type,
                 x_row_stride: tl.uint32,
                 H: tl.uint32,
                 eps: tl.float32,
                 BLOCK_SIZE: tl.constexpr):
    # Each program processes a row of the 'x' tensor.
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    # 'offsets' is used to index elements in the row.
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < H
    # Load the row elements using a mask since BLOCK_SIZE may be > H.
    x = tl.load(x_ptrs, mask=mask, other=0)
    # Compute variance
    x_mean_sq = tl.sum(x * x, axis=0) / H
    x_var = x_mean_sq + eps
    x_std = tl.sqrt(x_var)
    # Normalize and apply linear transformation
    x_hat = x / x_std
    output = x_hat * tl.load(weight_ptr + offsets, mask=mask, other=0)
    # Write output row-wise.
    output_ptrs = output_ptr + row_idx * x_row_stride + offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def rms_norm_backward(grad_output_ptr: tl.pointer_type,
                      grad_x_ptr: tl.pointer_type,
                      partial_grad_weight_ptr: tl.pointer_type,
                      x_ptr: tl.pointer_type,
                      weight_ptr: tl.pointer_type,
                      x_row_stride: tl.uint32,
                      H: tl.uint32,
                      eps: tl.float32,
                      BLOCK_SIZE: tl.constexpr):
    # Each program processes a row of the 'x' tensor.
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < H
    # Load the row elements using a mask since BLOCK_SIZE may be > H.
    x = tl.load(x_ptrs, mask=mask, other=0)
    grad_output = tl.load(grad_output_ptr + row_idx, mask=mask, other=0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0)
    # Compute variance
    x_mean_sq = tl.sum(x * x, axis=0) / H
    x_var = x_mean_sq + eps
    x_std = tl.sqrt(x_var)
    x_hat = x / x_std
    # grad_x (See Eq 10)
    grad_x = (grad_output * weight) * (1 / x_std) * (H - 1 - offsets / H * x_hat * x_hat)
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    # partial_grad_weight (See Eq 9)
    partial_grad_weight = grad_output * x_hat
    tl.store(partial_grad_weight_ptr + offsets, partial_grad_weight, mask=mask)

class RmsNormFuncTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        output = torch.empty_like(x)
        # The kernel only works for BLOCK_SIZE >= H, so we
        # set a block size equal to the highest power of two
        # smaller than H.
        BLOCK_SIZE = 2**(x.shape[1].bit_length() - 1)
        grid = (x.shape[0], )
        rms_norm_fwd[grid](x, weight, output, x.stride(0), x.shape[1], eps, BLOCK_SIZE)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # The kernel only works for BLOCK_SIZE >= H, so we
        # set a block size equal to the highest power of two
        # smaller than H.
        BLOCK_SIZE = 2**(x.shape[1].bit_length() - 1)
        # We launch N instances of the kernel, where N is the
        # first dimension of the input tensors.
        grid = (x.shape[0], )
        # The output tensors are already allocated for us.
        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(weight)
        rms_norm_backward[grid](grad_output,
                                grad_x,
                                partial_grad_weight,
                                x,
                                weight,
                                x.stride(0),
                                x.shape[1],
                                1e-6,
                                BLOCK_SIZE)
        # We return the gradients with respect to both input tensors,
        # and None for the optional fourth input tensor.
        return grad_x, partial_grad_weight, None

def rms_norm_triton(x, weight, eps):
    return RmsNormFuncTriton.apply(x, weight, eps)
