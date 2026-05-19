import torch
import triton
import triton.language as tl

class FusedHardShrinkDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p=0.5, training=True, inplace=False, lambd=0.5):
        if not training:
            return input

        assert 0.0 <= p <= 1.0, "Invalid dropout probability {}".format(p)

        if inplace:
            output = input
        else:
            output = input.clone()

        if input.is_floating_point():
            dtype = input.dtype
        else:
            dtype = torch.float32

        output_ = output.view(-1)
        n = output_.numel()
        output_mask = output_.abs() > lambd

        block_size = triton.next_power_of_2(max(1, math.ceil(math.sqrt(n))))
        max_grid_size = min(1 << 20, triton.cdiv(0x7fffffff, block_size))
        grid_size = min(triton.cdiv(n, block_size), max_grid_size)

        dropout_forward_kernel[grid_size,](
            output_, output_, n,
            p,
            seed=int(torch.randint(0x7fffffff, (1,)).item()),
            BLOCK_SIZE=block_size,
            dtype=dtype,
        )

        ctx.p = p
        ctx.training = training
        ctx.lambd = lambd
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, output_grad):
        if not ctx.training or not ctx.p:
            return output_grad, None, None, None, None

        output, = ctx.saved_tensors

        input_grad = output_grad.clone()

        output_grad_ = output_grad.view(-1)
        n = output_grad_.numel()
        output_mask = output_.abs() > ctx.lambd

        block_size = triton.next_power_of_2(max(1, math.ceil(math.sqrt(n))))
        max_grid_size = min(1 << 20, triton.cdiv(0x7fffffff, block_size))
        grid_size = min(triton.cdiv(n, block_size), max_grid_size)

        dropout_backward_kernel[grid_size,](
            output_grad_, input_grad, n,
            ctx.p,
            BLOCK_SIZE=block_size,
        )

        return input_grad, None, None, None, None

def fused_hardshrink_dropout(input, p=0.5, training=True, inplace=False, lambd=0.5):
    return FusedHardShrinkDropoutFunction.apply(input, p, training, inplace, lambd)
