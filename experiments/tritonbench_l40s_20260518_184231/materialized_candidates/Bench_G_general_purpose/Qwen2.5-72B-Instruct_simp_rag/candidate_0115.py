import triton
import triton.language as tl

@triton.jit
def rms_norm_fwd(x_ptr: tl.pointer_type,
                 weight_ptr: tl.pointer_type,
                 x_row_stride: tl.uint32,
                 output_ptr: tl.pointer_type,
                 H: tl.uint32,
                 eps: tl.float32,
                 BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    # Load input row and gain
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    gain = tl.load(weight_ptr + offsets, mask=mask, other=1)

    # Compute RMS
    squared_row = x_row * x_row
    squared_mean = tl.sum(squared_row) / H
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and apply gain
    normalized_row = x_row / rms
    scaled_row = normalized_row * gain

    # Store the result in the output
    tl.store(output_ptr + row_idx * x_row_stride + offsets, scaled_row, mask=mask)

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
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    grad_output_row = tl.load(grad_output_ptr + row_idx * x_row_stride + offsets, mask=mask, other=0)
    x_row = tl.load(x_ptr + row_idx * x_row_stride + offsets, mask=mask, other=0)
    gain_row = tl.load(weight_ptr + offsets, mask=mask, other=1)

    squared_row = tl.sum(x_row * x_row)
    rms = tl.sqrt(squared_row / H + eps)

    normalized_row = x_row / rms
    grad_x = (grad_output_row * gain_row) / rms

    grad_x += - x_row * tl.sum(grad_x * x_row) / (rms * rms * H)
    tl.store(grad_x_ptr + row_idx * x_row_stride + offsets, grad_x, mask=mask)

    grad_gain_row = grad_output_row * normalized_row
    tl.store(partial_grad_weight_ptr + row_idx * x_row_stride + offsets, grad_gain_row, mask=mask)

import torch

class RMS_Norm_Func_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Remember x and weight for the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight)

        H = x.shape[-1]
        n_rows = x.numel() // H  # Flatten other dimensions
        x_reshaped = x.reshape(n_rows, H)

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)

        y_reshaped = torch.empty((n_rows, H), device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        rms_norm_fwd[(n_rows,)](
            x, weight, x_reshaped.stride(0), y_reshaped, H, eps=1e-9,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        y = y_reshaped.view(x.shape)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors

        H = x.shape[-1]
        n_rows = x.numel() // H  # Flatten other dimensions
        x_reshaped = x.reshape(n_rows, H)

        partial_grad_weight = torch.empty_like(x_reshaped)
        grad_x = torch.empty_like(x_reshaped)
        rms_norm_backward[(n_rows,)](
            grad_out, grad_x, partial_grad_weight,
            x_reshaped, weight, x_reshaped.stride(0), H, 1e-5,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return grad_x.view(x.shape), partial_grad_weight.sum(axis=0)

# Example usage
x = torch.randn((2, 3, 4), device='cuda')
weight = torch.randn(4, device='cuda', requires_grad=True)

y = RMS_Norm_Func_Triton.apply(x, weight)
y.backward(torch.ones_like(y))

print("Input x:", x)
print("Weight:", weight)
print("Output y:", y)
print("Gradient of weight:", weight.grad)
