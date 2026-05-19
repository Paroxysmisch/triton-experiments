, each processing
        # one row of x.
        grid = (triton.cdiv(H, ctx.BLOCK_SIZE), )

        weighted_sum_fwd[grid](x, weight, x.stride(-2), y, H, ctx.BLOCK_SIZE)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        assert grad_output.is_cuda and x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert grad_output.is_contiguous(), "Our pointer arithmetic will assume contiguous grad_output"

        output_dims = grad_output.shape[:-1]
        H = x.shape[-1]
        grad_output = grad_output.reshape(-1)
        x = x.reshape(-1, H)
        weight = weight.reshape(-1, H)
        n_rows = grad_output.numel() // grad_output.size(-1)

        assert x.shape[0] == n_rows, "Dimension mismatch"
        assert weight.shape[0] == n_rows, "Dimension mismatch"

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(weight)

        # Write a kernel that can process multiple rows simultaneously.
        BLOCK_SIZE = triton.next_power_of_2(H)

        def grid(meta):
            return (triton.cdiv(H, meta["BLOCK_SIZE"]), n_rows)

        weighted_sum_backward[grid](grad_output, grad_x, partial_grad_weight, x, weight, x.stride(-2), H, BLOCK_SIZE)
        partial_grad_weight = partial_grad_weight.sum(0).view(weight.shape)
        return grad_x.reshape(output_dims + x.shape[-2:]), partial_grad_weight, None

def weighted_sum_triton(x, weight):
    return WeightedSumFunc_Triton.apply(x, weight)

@triton.jit
def rms_norm_fwd(x_ptr, weight_ptr, output_ptr, x_row_stride, H, EPS, BLOCK_SIZE):
    # 1. Compute the RMS normalization.
    #    This is equivalent to layer norm but without the affine transform
    #    and the mean (only the root-mean-square).
    # 2. Multiply the normalized value by the gain 'gamma', which is just
    #    a learned weight.
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < H
    # Root mean square (RMS) normalization.
    x = tl.load(x_ptrs, mask=mask, other=0)
    x_gathered = tl.where(mask, x, 0.)
    rms = tl.sqrt(tl.sum(x_gathered * x_gathered, axis=0) / H + EPS)
    output = x / rms
    # Store the output in the appropriate position in the output tensor.
    output_ptrs = output_ptr + row_idx * x_row_stride
    tl.store(output_ptrs + offsets, output, mask=mask)

@triton.jit
def rms_norm_backward(
        grad_output_ptr,
        grad_x_ptr,
        partial_grad_weight_ptr,
        x_ptr,
        weight_ptr,
        x_row_stride,
        H,
        EPS,
        BLOCK_SIZE
):
    # See forward pass for a more detailed explanation.
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    x_ptrs = row_start_ptr + offsets
    grad_output_ptrs = weight_ptr + offsets
    x = tl.load(x_ptrs, mask=mask, other=0)
    grad_output = tl.load(grad_output_ptr + row_idx)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / H + EPS)
    grad_x = (grad_output * (weight_ptr + offsets)).sum() / rms
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    partial_grad_weight = (grad_output * x / rms).to(weight_ptr.dtype.element_ty)
    tl.store(partial_grad_weight_ptr + offsets, partial_grad_weight, mask=mask)

class RmsNormFunc_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCK_SIZE"]), )
        rms_norm_fwd[grid](x, weight, output, x.stride(0), x.shape[1], eps, BLOCK_SIZE=1024)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(weight)

        n_rows = grad_output.numel() // grad_output.size(-1)
        BLOCK_SIZE = triton.next_power_of_2(x.shape[1])

        def grid(meta):
            return (triton.cdiv(x.shape[1], meta["BLOCK_SIZE"]), n_rows)

        rms_norm_backward[grid](grad_output, grad_x, partial_grad_weight, x, weight, x.stride(0), x.shape[1], eps,
                                 BLOCK_SIZE)
        partial_grad_weight = partial_grad_weight.sum(0).view(weight.shape)
        return grad_x, partial_grad_weight, None

def rms_norm_triton(x, weight, eps):
    return RmsNormFunc_Triton.apply(x, weight, eps)
