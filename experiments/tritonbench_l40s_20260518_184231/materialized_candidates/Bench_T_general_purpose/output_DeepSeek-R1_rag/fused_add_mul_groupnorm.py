import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm_kernel(
    input1_ptr, input2_ptr, output_ptr, gamma_ptr, beta_ptr,
    num_channels, num_groups, eps,
    N, C, S,
    stride_input1_n, stride_input1_c, stride_input1_s,
    stride_input2_n, stride_input2_c, stride_input2_s,
    stride_output_n, stride_output_c, stride_output_s,
    stride_gamma_c, stride_beta_c,
    BLOCK_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    c_start = g * channels_per_group
    group_size = channels_per_group * S

    sum = 0.0
    sum_sq = 0.0

    for i in range(0, group_size, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        c_in_group = idx // S
        s = idx % S
        c = c_start + c_in_group

        off_input1 = n * stride_input1_n + c * stride_input1_c + s * stride_input1_s
        off_input2 = n * stride_input2_n + c * stride_input2_c + s * stride_input2_s

        x = tl.load(input1_ptr + off_input1, mask=mask, other=0.0)
        y = tl.load(input2_ptr + off_input2, mask=mask, other=0.0)
        z = x + y
        m = z * y

        sum += tl.sum(m, mask=mask)
        sum_sq += tl.sum(m * m, mask=mask)

    mean = sum / group_size
    var = (sum_sq / group_size) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    for i in range(0, group_size, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        c_in_group = idx // S
        s = idx % S
        c = c_start + c_in_group

        off_input1 = n * stride_input1_n + c * stride_input1_c + s * stride_input1_s
        off_input2 = n * stride_input2_n + c * stride_input2_c + s * stride_input2_s

        x = tl.load(input1_ptr + off_input1, mask=mask, other=0.0)
        y = tl.load(input2_ptr + off_input2, mask=mask, other=0.0)
        z = x + y
        m = z * y

        normalized = (m - mean) * inv_std
        gamma = tl.load(gamma_ptr + c * stride_gamma_c, mask=mask)
        beta = tl.load(beta_ptr + c * stride_beta_c, mask=mask)
        out = normalized * gamma + beta

        off_output = n * stride_output_n + c * stride_output_c + s * stride_output_s
        tl.store(output_ptr + off_output, out, mask=mask)

class FusedAddMulGroupnormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, weight, bias, num_groups, eps):
        assert input1.shape == input2.shape, "input2 must be broadcastable to input1's shape"
        C = input1.size(1)
        assert weight.shape == (C,), f"Weight shape {weight.shape} != ({C},)"
        assert bias.shape == (C,), f"Bias shape {bias.shape} != ({C},)"
        assert C % num_groups == 0, f"{num_groups} must divide {C}"

        N, C = input1.shape[0], input1.shape[1]
        S = input1.numel() // (N * C)
        input1_3d = input1.contiguous().view(N, C, S)
        input2_3d = input2.expand_as(input1).contiguous().view(N, C, S)
        output = torch.empty_like(input1_3d)

        BLOCK_SIZE = 512
        grid = (N, num_groups)
        fused_add_mul_groupnorm_kernel[grid](
            input1_3d, input2_3d, output, weight, bias,
            C, num_groups, eps,
            N, C, S,
            input1_3d.stride(0), input1_3d.stride(1), input1_3d.stride(2),
            input2_3d.stride(0), input2_3d.stride(1), input2_3d.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            weight.stride(0), bias.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(input1, input2, weight, bias)
        ctx.num_groups, ctx.eps = num_groups, eps
        return output.view_as(input1)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, weight, bias = ctx.saved_tensors
        num_groups, eps = ctx.num_groups, ctx.eps

        with torch.enable_grad():
            input1_ = input1.detach().requires_grad_(True)
            input2_ = input2.detach().requires_grad_(True)
            weight_ = weight.detach().requires_grad_(True)
            bias_ = bias.detach().requires_grad_(True)

            z = input1_ + input2_
            m = z * input2_
            m_ = m.contiguous()
            output_ = F.group_norm(m_, num_groups, weight_, bias_, eps)
            output_.backward(grad_output, retain_graph=True)

        return input1_.grad, input2_.grad, weight_.grad, bias_.grad, None, None

def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, out=None):
    result = FusedAddMulGroupnormFunction.apply(input1, input2, weight, bias, num_groups, eps)
    if out is not None:
        out.copy_(result)
        return out
    return result
