import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    X, Y, M, W, B, O,
    stride_xb, stride_xc, stride_yb, stride_yc, stride_mb, stride_mc, stride_ob, stride_oc,
    stride_w, stride_b,
    N, C, H, W, num_groups, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W

    # Load data
    x_offsets = (offsets // (C * H * W)) * stride_xb + (offsets % (C * H * W)) % (H * W) + (offsets % (C * H * W)) // (H * W) * stride_xc
    y_offsets = (offsets // (C * H * W)) * stride_yb + (offsets % (C * H * W)) % (H * W) + (offsets % (C * H * W)) // (H * W) * stride_yc
    x = tl.load(X + x_offsets, mask=mask)
    y = tl.load(Y + y_offsets, mask=mask)

    # Element-wise addition and multiplication
    z = x + y
    m = z * y

    # Group normalization
    group_size = C // num_groups
    group_id = (offsets % (C * H * W)) // (group_size * H * W)
    group_start = group_id * group_size * H * W
    group_end = group_start + group_size * H * W

    # Compute mean and variance within the group
    group_mask = (offsets % (C * H * W) >= group_start) & (offsets % (C * H * W) < group_end)
    group_mean = tl.sum(m, mask=group_mask) / (group_size * H * W)
    group_var = tl.sum((m - group_mean) ** 2, mask=group_mask) / (group_size * H * W)
    inv_std = 1 / tl.sqrt(group_var + eps)

    # Apply group normalization
    w_offsets = (offsets % (C * H * W)) // (H * W) * stride_w
    b_offsets = (offsets % (C * H * W)) // (H * W) * stride_b
    w = tl.load(W + w_offsets, mask=mask)
    b = tl.load(B + b_offsets, mask=mask)
    o = (m - group_mean) * inv_std * w + b

    # Store the result
    o_offsets = (offsets // (C * H * W)) * stride_ob + (offsets % (C * H * W)) % (H * W) + (offsets % (C * H * W)) // (H * W) * stride_oc
    tl.store(O + o_offsets, o, mask=mask)

import torch
import triton
import triton.language as tl

class FusedAddMulGroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, W, B, num_groups, eps=1e-5, out=None):
        N, C, H, W = X.shape
        assert Y.shape == X.shape, "Input tensors must be broadcastable to each other"
        assert W.shape == (C,), "Weight parameter must have shape (C,)"
        assert B.shape == (C,), "Bias parameter must have shape (C,)"
        assert C % num_groups == 0, "Number of groups must divide the number of channels evenly"

        if out is None:
            out = torch.empty_like(X)

        # Launch the Triton kernel
        grid = (N * C * H * W // 1024 + 1,)
        fused_add_mul_groupnorm_kernel[grid](
            X, Y, out, W, B, out,
            X.stride(0), X.stride(1), Y.stride(0), Y.stride(1), out.stride(0), out.stride(1), out.stride(0), out.stride(1),
            W.stride(0), B.stride(0),
            N, C, H, W, num_groups, eps, 1024
        )

        ctx.save_for_backward(X, Y, W, B, out, num_groups, eps)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, W, B, out, num_groups, eps = ctx.saved_tensors
        N, C, H, W = X.shape

        # Compute gradients
        grad_X = torch.zeros_like(X)
        grad_Y = torch.zeros_like(Y)
        grad_W = torch.zeros_like(W)
        grad_B = torch.zeros_like(B)

        # Reuse the forward pass kernel to compute gradients
        grid = (N * C * H * W // 1024 + 1,)
        fused_add_mul_groupnorm_kernel[grid](
            X, Y, out, W, B, grad_X,
            X.stride(0), X.stride(1), Y.stride(0), Y.stride(1), grad_X.stride(0), grad_X.stride(1), grad_X.stride(0), grad_X.stride(1),
            W.stride(0), B.stride(0),
            N, C, H, W, num_groups, eps, 1024
        )

        # Compute gradients for Y
        grad_Y = grad_output * (X + Y) + grad_output * (X + Y) * Y

        # Compute gradients for W and B
        group_size = C // num_groups
        for group_id in range(num_groups):
            group_start = group_id * group_size
            group_end = group_start + group_size
            group_mask = (torch.arange(C) >= group_start) & (torch.arange(C) < group_end)
            group_mean = torch.mean(out[:, group_mask, :, :], dim=(1, 2, 3), keepdim=True)
            group_var = torch.var(out[:, group_mask, :, :], dim=(1, 2, 3), keepdim=True)
            inv_std = 1 / torch.sqrt(group_var + eps)

            grad_W[group_mask] = torch.sum(grad_output[:, group_mask, :, :] * (out[:, group_mask, :, :] - group_mean) * inv_std, dim=(0, 2, 3))
            grad_B[group_mask] = torch.sum(grad_output[:, group_mask, :, :], dim=(0, 2, 3))

        return grad_X, grad_Y, grad_W, grad_B, None, None, None

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
    return FusedAddMulGroupNorm.apply(input1, input2, weight, bias, num_groups, eps, out)
