import torch
import triton
import triton.language as tl

@triton.jit
def fused_bmm_dropout_gelu_forward_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    stride_b1, stride_n1, stride_m1,
    stride_b2, stride_m2, stride_p2,
    stride_b_out, stride_n_out, stride_p_out,
    drop_p, seed, gelu_approximate,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)

    if pid_batch >= B or pid_n >= N or pid_p >= P:
        return

    acc = 0.0
    for m in range(0, M):
        offset1 = pid_batch * stride_b1 + pid_n * stride_n1 + m * stride_m1
        a = tl.load(input1_ptr + offset1)
        offset2 = pid_batch * stride_b2 + m * stride_m2 + pid_p * stride_p2
        b = tl.load(input2_ptr + offset2)
        acc += a * b

    element_offset = pid_batch * N * P + pid_n * P + pid_p
    random = tl.rand(seed, element_offset)
    scale = 1.0 / (1.0 - drop_p) if drop_p < 1.0 else 1.0
    mask = random >= drop_p
    dropped = acc * scale * tl.where(mask, 1.0, 0.0)

    if gelu_approximate == 0:
        gelu = dropped * 0.5 * (1.0 + tl.erf(dropped / tl.sqrt(2.0)))
    else:
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        x = dropped
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + coeff * x_cubed)
        tanh_inner = tl.tanh(inner)
        gelu = 0.5 * x * (1 + tanh_inner)

    offset_out = pid_batch * stride_b_out + pid_n * stride_n_out + pid_p * stride_p_out
    tl.store(output_ptr + offset_out, gelu)

class FusedBmmDropoutGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, drop_p, training, inplace, approximate):
        B, N, M = input1.shape
        B2, M2, P = input2.shape
        assert B == B2 and M == M2, "Input dimensions mismatch for BMM"

        output = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)

        effective_drop_p = drop_p if training else 0.0
        max_seed = 2**32 - 1
        seed = torch.randint(0, max_seed, (1,), device=input1.device).item()
        gelu_approximate_flag = 0 if approximate == 'none' else 1

        grid = (B, N, P)
        fused_bmm_dropout_gelu_forward_kernel[grid](
            input1, input2, output,
            B, N, M, P,
            input1.stride(0), input1.stride(1), input1.stride(2),
            input2.stride(0), input2.stride(1), input2.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            effective_drop_p, seed, gelu_approximate_flag,
            BLOCK_SIZE=128
        )

        ctx.save_for_backward(input1, input2)
        ctx.effective_drop_p = effective_drop_p
        ctx.seed = seed
        ctx.gelu_approximate = gelu_approximate_flag
        ctx.training = training

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        effective_drop_p = ctx.effective_drop_p
        seed = ctx.seed
        gelu_approximate = ctx.gelu_approximate
        training = ctx.training

        grad_input1 = grad_input2 = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            raise NotImplementedError("Backward pass is not implemented for this fused kernel. "
                                      "Please use separate PyTorch operations for gradient computation.")

        return grad_input1, grad_input2, None, None, None, None, None

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    if inplace:
        raise NotImplementedError("Inplace operation is not supported for fused_bmm_dropout_gelu")
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")

    input1 = input1.contiguous()
    input2 = input2.contiguous()

    output = FusedBmmDropoutGeluFunction.apply(input1, input2, p, training, inplace, approximate)

    if out is not None:
        out.copy_(output)
        return out
    return output
