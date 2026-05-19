import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(X, W, Y, stride_x, stride_y, stride_w, stride_out, N):
    # Pointers to the start of the row in each tensor
    row_id = tl.program_id(0)
    X_row_ptr = X + row_id * stride_x
    W_row_ptr = W + row_id * stride_w
    Y_row_ptr = Y + row_id * stride_y

    # Load the row
    x = tl.load(X_row_ptr + tl.arange(0, N))
    
    # Compute RMS
    mean_sq = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(mean_sq)

    # Normalize
    x_norm = x / rms

    # Scale by weight
    w = tl.load(W_row_ptr + tl.arange(0, N))
    y = x_norm * w

    # Store the result
    tl.store(Y_row_ptr + tl.arange(0, N), y)

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        assert x.is_contiguous() and w.is_contiguous()
        assert x.shape[1] == w.shape[0]

        # Allocate output
        y = torch.empty_like(x)

        # Grid
        num_rows = x.shape[0]
        N = x.shape[1]
        grid = (num_rows,)

        # Launch kernel
        rms_norm_kernel[grid](
            x, w, y,
            x.stride(0), y.stride(0), w.stride(0), y.stride(0),
            N
        )

        # Save for backward
        ctx.save_for_backward(x, w, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, w, y = ctx.saved_tensors
        N = x.shape[1]

        # Compute gradients
        grad_x = torch.empty_like(x)
        grad_w = torch.empty_like(w)

        # Calculate RMS and normalized x
        mean_sq = torch.sum(x * x, dim=1, keepdim=True) / N
        rms = torch.sqrt(mean_sq)
        x_norm = x / rms

        # Compute grad_w
        grad_w = torch.sum(grad_output * x_norm, dim=0)

        # Compute grad_x
        grad_x_norm = grad_output * w
        grad_rms = -torch.sum(grad_x_norm * x_norm, dim=1, keepdim=True) / rms
        grad_x = grad_x_norm / rms + (x * grad_rms) / (N * rms)

        return grad_x, grad_w

# Example usage
x = torch.randn(10, 5, device='cuda', requires_grad=True)
w = torch.randn(5, device='cuda', requires_grad=True)
y = RmsNorm.apply(x, w)
