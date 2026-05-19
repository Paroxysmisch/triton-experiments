import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   x_row_stride: tl.uint32,
                   output_ptr: tl.pointer_type,
                   H: tl.uint32,
                   eps: tl.float32,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    # Load input row and weight
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    rms_w = tl.load(rms_w_ptr + offsets, mask=mask, other=1)

    # Compute RMS
    squared_row = x_row * x_row
    squared_mean = tl.sum(squared_row) / H
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and apply weight
    normalized_row = x_row / rms
    scaled_row = normalized_row * rms_w

    # Store the result in the output
    tl.store(output_ptr + row_idx * x_row_stride + offsets, scaled_row, mask=mask)

import torch

class RMSNormFuncTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rms_w):
        # Remember x and rms_w for the backward pass
        ctx.save_for_backward(x, rms_w)

        B, M, H = x.shape
        assert len(rms_w.shape) == 1 and rms_w.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and rms_w.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        output = torch.empty_like(x)

        # Launch our kernel with B * M instances in our 1D grid.
        rmsnorm_triton[(B * M,)](
            x, rms_w, x.stride(1), output, H, eps=1e-9,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, rms_w = ctx.saved_tensors

        B, M, H = x.shape
        partial_grad_rms_w = torch.empty_like(x)
        grad_x = torch.empty_like(x)

        rmsnorm_triton_backward[(B * M,)](
            grad_out, grad_x, partial_grad_rms_w,
            x, rms_w, x.stride(1), H, 1e-5,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return grad_x, partial_grad_rms_w.sum(dim=(0, 1))

@triton.jit
def rmsnorm_triton_backward(grad_output_ptr: tl.pointer_type,
                            grad_x_ptr: tl.pointer_type,
                            partial_grad_rms_w_ptr: tl.pointer_type,
                            x_ptr: tl.pointer_type,
                            rms_w_ptr: tl.pointer_type,
                            x_row_stride: tl.uint32,
                            H: tl.uint32,
                            eps: tl.float32,
                            BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    grad_output_row = tl.load(grad_output_ptr + row_idx * x_row_stride + offsets, mask=mask, other=0)
    x_row = tl.load(x_ptr + row_idx * x_row_stride + offsets, mask=mask, other=0)
    rms_w_row = tl.load(rms_w_ptr + offsets, mask=mask, other=1)

    squared_row = tl.sum(x_row * x_row)
    rms = tl.sqrt(squared_row / H + eps)

    normalized_row = x_row / rms
    grad_x = (grad_output_row * rms_w_row) / rms

    grad_x += - x_row * tl.sum(grad_x * x_row) / (rms * rms * H)
    tl.store(grad_x_ptr + row_idx * x_row_stride + offsets, grad_x, mask=mask)

    grad_rms_w_row = grad_output_row * normalized_row
    tl.store(partial_grad_rms_w_ptr + row_idx * x_row_stride + offsets, grad_rms_w_row, mask=mask)
