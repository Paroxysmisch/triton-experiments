)
        # Weighted sum's backward pass does not require the gradients
        # wrt. the weight vector, so we return None for it.
        return grad_x, None, partial_grad_weight.sum(0), None

def weighted_sum_triton(x, weight):
    return WeightedSumFunc_Triton.apply(x, weight)

@triton.jit
def rms_norm_fwd(x_ptr: tl.pointer_type,
                 rms_w_ptr: tl.pointer_type,
                 output_ptr: tl.pointer_type,
                 x_row_stride: tl.uint32,
                 N: tl.uint32,
                 eps: tl.float32,
                 BLOCK_SIZE: tl.constexpr):
    # Each program instance will process one row of the input.
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < N
    # The stride is the distance we need to travel in memory to get to the next row.
    # If we are at the end of a row, moving forward in memory should take us to the
    # start of the next row.
    next_row_start_ptr = row_start_ptr + N
    row_stride = next_row_start_ptr - row_start_ptr
    # The block size is the next power of two greater than N, so we know that
    # the last dimension of the input will always be BLOCK_SIZE.
    # We can therefore load the entire row into a "block", which is just a 1D tensor
    # of size BLOCK_SIZE, since the stride is guaranteed to be 0.
    x_block_ptr = tl.make_block_ptr(base=x_ptrs, shape=(BLOCK_SIZE, ), strides=(1, ), offsets=(0, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    # Load the row using the block pointer.
    row = tl.load(x_block_ptr, boundary_check=(0, ), padding_option='zero')
    # The mask will be zero for invalid entries, and one for valid entries.
    # This is useful for example if the last dimension of the input is not a power of two.
    row = tl.where(mask, row, 0)
    # Compute variance
    row_var = tl.sum(row * row, axis=0) / N
    row_rms = tl.sqrt(row_var + eps)
    # Normalize, apply rms weight, and write back output
    norm = tl.fdiv(row, row_rms)
    rms_weight = tl.load(rms_w_ptr + offsets, mask=mask, other=0)
    output = norm * rms_weight
    output_ptrs = output_ptr + row_idx * x_row_stride + offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def rms_norm_backward(grad_output_ptr: tl.pointer_type,
                      grad_x_ptr: tl.pointer_type,
                      partial_grad_rms_w_ptr: tl.pointer_type,
                      x_ptr: tl.pointer_type,
                      rms_w_ptr: tl.pointer_type,
                      x_row_stride: tl.uint32,
                      N: tl.uint32,
                      eps: tl.float32,
                      BLOCK_SIZE: tl.constexpr):
    # Each program instance handles one row of the input.
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < N
    next_row_start_ptr = row_start_ptr + N
    row_stride = next_row_start_ptr - row_start_ptr
    # As always, we can load the entire row into a "block" using a block pointer.
    x_block_ptr = tl.make_block_ptr(base=x_ptrs, shape=(BLOCK_SIZE, ), strides=(1, ), offsets=(0, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    row = tl.load(x_block_ptr, boundary_check=(0, ), padding_option='zero')
    # The mask will be zero for invalid entries, and one for valid entries.
    # This is useful for example if the last dimension of the input is not a power of two.
    row = tl.where(mask, row, 0)
    # Compute variance
    row_var = tl.sum(row * row, axis=0) / N
    row_rms = tl.sqrt(row_var + eps)
    # Compute partial gradients
    # See the paper for more information on how these were derived.
    # grad_rms_w is the gradient of the loss w.r.t. the rms weight.
    grad_output = tl.load(grad_output_ptr + row_idx)
    norm = tl.fdiv(row, row_rms)
    rms_weight = tl.load(rms_w_ptr + offsets, mask=mask, other=0)
    grad_rms_w = grad_output * norm
    partial_grad_rms_w_ptr = partial_grad_rms_w_ptr + row_idx * x_row_stride + offsets
    tl.store(partial_grad_rms_w_ptr, grad_rms_w, mask=mask)
    grad_x = grad_output * rms_weight * tl.fdiv(N, row_rms) * (1. + offsets * 0.)
    grad_x_block_ptr = tl.make_block_ptr(base=grad_x_ptr + row_idx * x_row_stride,
                                         shape=(N, ), strides=(1, ), offsets=(0, ),
                                         block_shape=(BLOCK_SIZE, ), order=(0, ))
    tl.store(grad_x_block_ptr, grad_x, boundary_check=(0, ), padding_option='zero')

class RmsNormFunc_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rms_w, eps):
        ctx.save_for_backward(x, rms_w)
        output = torch.empty_like(x)
        x = x.contiguous()
        assert x.is_cuda and rms_w.is_cuda
        M, N = x.shape
        grid = (M, )
        rms_norm_fwd[grid](x, rms_w, output, x.stride(0), N, eps, BLOCK_SIZE=triton.next_power_of_2(N))
        output[:, :] = output[:, :]
        ctx.BLOCK_SIZE = triton.next_power_of_2(N)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, rms_w = ctx.saved_tensors
        partial_grad_rms_w = torch.empty_like(x)
        grad_x = torch.empty_like(x)
        M, N = x.shape
        rms_norm_backward[(M, )](grad_output, grad_x, partial_grad_rms_w, x, rms_w, x.stride(0), N, ctx.eps,
                                  BLOCK_SIZE=ctx.BLOCK_SIZE)
        # As explained in the paper, we do not backpropagate through rms_w.
        return grad_x, None, partial_grad_rms_w.sum(0), None

def rms_norm_triton(x, rms_w, eps=1e-6):
    return RmsNormFunc_Triton.apply(x, rms_w, eps)
