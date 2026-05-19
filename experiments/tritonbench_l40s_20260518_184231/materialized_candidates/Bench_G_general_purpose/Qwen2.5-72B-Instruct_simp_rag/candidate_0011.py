import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe(x_ptr: tl.pointer_type,
                   weight_ptr: tl.pointer_type,
                   output_ptr: tl.pointer_type,
                   x_row_stride: tl.uint32,
                   weight_row_stride: tl.uint32,
                   output_row_stride: tl.uint32,
                   H: tl.uint32,
                   N: tl.uint32,
                   M: tl.uint32,
                   eps: tl.float32,
                   apply_rbe: tl.int32,
                   rbe_theta: tl.float32,
                   rbe_idx: tl.pointer_type,
                   BLOCK_SIZE: tl.constexpr):
    # Each instance will compute the matrix multiplication for a row of x.
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return

    # Pointer to the first entry of the row this instance sums up.
    row_start_ptr = x_ptr + row_idx * x_row_stride
    weight_start_ptr = weight_ptr
    output_start_ptr = output_ptr + row_idx * output_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H

    # Load input row and weight
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    weight_row = tl.load(weight_start_ptr + offsets, mask=mask, other=0)

    # Compute RMS
    squared_row = x_row * x_row
    squared_mean = tl.sum(squared_row) / H
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and apply gain (weight)
    normalized_row = x_row / rms
    scaled_row = normalized_row * weight_row

    # Apply Rotary Embeddings if required
    if apply_rbe:
        rbe_idx_row = tl.load(rbe_idx + row_idx * H + offsets, mask=mask, other=0)
        cos_theta = tl.cos(rbe_theta * rbe_idx_row)
        sin_theta = tl.sin(rbe_theta * rbe_idx_row)
        x_rotated = normalized_row * cos_theta + tl.roll(normalized_row, 1) * sin_theta
        scaled_row = x_rotated * weight_row

    # Compute the dot product
    output = tl.sum(scaled_row)

    # Write back output (a single scalar per instance).
    tl.store(output_start_ptr, output)

class RMS_Matmul_RBE_Func_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, rbe_theta, rbe_idx, apply_rbe):
        # Remember x and weight for the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight, rbe_theta, rbe_idx, apply_rbe)

        N, H = x.shape
        M, _ = weight.shape

        assert len(weight.shape) == 2 and weight.shape[1] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and weight.is_contiguous(), "Our pointer arithmetic will assume contiguous tensors"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty((N, M), device=x.device)

        # Launch our kernel with N instances in our 1D grid.
        rms_matmul_rbe[(N,)](
            x, weight, y, x.stride(0), weight.stride(0), y.stride(0), H, N, M, 1e-9, apply_rbe, rbe_theta, rbe_idx,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rbe_theta, rbe_idx, apply_rbe = ctx.saved_tensors
        N, H = x.shape
        M, _ = weight.shape

        # Allocate output tensors.
        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)

        # Launch our kernel with N instances in our 1D grid.
        rms_matmul_rbe[(N,)](
            grad_out, weight, grad_x, x.stride(0), weight.stride(0), grad_x.stride(0), H, N, M, 1e-9, apply_rbe, rbe_theta, rbe_idx,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)

        # Compute the gradient for the weight
        for i in range(M):
            weight_grad = torch.sum(grad_out[:, i].unsqueeze(-1) * x, dim=0)
            grad_weight[i] = weight_grad

        return grad_x, grad_weight, None, None, None

# Example usage
if __name__ == "__main__":
    N, H, M = 1024, 128, 64
    x = torch.randn((N, H), device="cuda")
    weight = torch.randn((M, H), device="cuda")
    rbe_theta = 0.01
    rbe_idx = torch.arange(N * H, device="cuda").view(N, H)
    apply_rbe = 1

    y = RMS_Matmul_RBE_Func_Triton.apply(x, weight, rbe_theta, rbe_idx, apply_rbe)
    print(y)
