import torch
import triton
import triton.language as tl

@triton.jit
def rms_layernorm_forward_kernel(
    X, Y, mean, inv_std, N, eps, BLOCK_SIZE: tl.constexpr
):
    # Compute the block index
    block_idx = tl.program_id(axis=0)
    # Define the range of elements this block will process
    start = block_idx * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, N)
    # Load the input data for this block
    x = tl.load(X + start, mask=start < N)
    # Compute mean
    mean_val = tl.sum(x, axis=0) / N
    # Compute variance and inverse standard deviation
    var = tl.sum((x - mean_val) ** 2, axis=0) / N
    inv_std_val = 1 / tl.sqrt(var + eps)
    # Normalize
    y = (x - mean_val) * inv_std_val
    # Store results
    tl.store(Y + start, y, mask=start < N)
    tl.store(mean + block_idx, mean_val)
    tl.store(inv_std + block_idx, inv_std_val)

@triton.jit
def rms_layernorm_backward_kernel(
    dY, X, dX, mean, inv_std, N, BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(axis=0)
    start = block_idx * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, N)
    # Load necessary data
    dy = tl.load(dY + start, mask=start < N)
    x = tl.load(X + start, mask=start < N)
    mean_val = tl.load(mean + block_idx)
    inv_std_val = tl.load(inv_std + block_idx)
    # Compute gradients
    dx = inv_std_val * (dy - tl.sum(dy, axis=0) / N - (x - mean_val) * tl.sum(dy * (x - mean_val), axis=0) / (N * inv_std_val ** 2))
    # Store gradient
    tl.store(dX + start, dx, mask=start < N)

class FastRMSLayernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps=1e-5):
        N = input.shape[-1]
        output = torch.empty_like(input)
        mean = torch.empty(input.shape[0], device=input.device)
        inv_std = torch.empty(input.shape[0], device=input.device)

        # Launch the forward kernel
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        rms_layernorm_forward_kernel[grid](
            input, output, mean, inv_std, N, eps, BLOCK_SIZE=1024
        )

        # Save for backward
        ctx.save_for_backward(input, mean, inv_std)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, inv_std = ctx.saved_tensors
        N = input.shape[-1]
        grad_input = torch.empty_like(input)

        # Launch the backward kernel
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        rms_layernorm_backward_kernel[grid](
            grad_output, input, grad_input, mean, inv_std, N, BLOCK_SIZE=1024
        )

        return grad_input, None

def fast_rms_layernorm(input, eps=1e-5):
    return FastRMSLayernorm.apply(input, eps)

# Simple layernorm module for testing
class SimpleLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(SimpleLayerNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input):
        return self.weight * fast_rms_layernorm(input, self.eps) + self.bias
