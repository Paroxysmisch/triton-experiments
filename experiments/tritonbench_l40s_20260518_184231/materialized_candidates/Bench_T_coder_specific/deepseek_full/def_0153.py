import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import JITFunction

@triton.jit
def adaptive_avg_pool2d_kernel(
    X,
    Y,
    output_size_h,
    output_size_w,
    input_size_h,
    input_size_w,
    num_output_features,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    input_row_start = pid_n * num_output_features
    output_row_start = pid_n * num_output_features

    output_offset = output_row_start * output_size_h * output_size_w + tl.arange(0, BLOCK_H)[:, None] * output_size_w + tl.arange(0, BLOCK_W)[None, :]
    output_mask = (output_row_start * output_size_h * output_size_w + tl.arange(0, BLOCK_H)[:, None] * output_size_w + tl.arange(0, BLOCK_W)[None, :] < num_output_features * output_size_h * output_size_w)

    input_offset = input_row_start * input_size_h * input_size_w + tl.arange(0, BLOCK_H)[:, None] * input_size_w + tl.arange(0, BLOCK_W)[None, :]
    input_mask = (input_row_start * input_size_h * input_size_w + tl.arange(0, BLOCK_H)[:, None] * input_size_w + tl.arange(0, BLOCK_W)[None, :] < num_output_features * input_size_h * input_size_w)

    block_output = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    block_input = tl.load(X + input_offset, mask=input_mask, other=0.0)
    block_output += block_input

    num_elements = tl.sum(input_mask.to(tl.float32))
    block_output = block_output / (num_elements + 1e-5)

    tl.store(Y + output_offset, block_output, mask=output_mask)


class AdaptiveAvgPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size):
        if input.dim() == 3:
            batch_size, channels, num_features = input.size()
            num_batches = 1
        elif input.dim() == 4:
            batch_size, channels, num_batches, num_features = input.size()
        else:
            raise ValueError("Invalid input shape, expected 3D or 4D tensor")

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif len(output_size) == 1:
            output_size = (output_size[0], output_size[0])
        else:
            raise ValueError("Invalid output_size")

        output_size_h, output_size_w = output_size
        num_output_features = channels

        output = torch.empty(batch_size, num_output_features, output_size_h, output_size_w, device=input.device, dtype=input.dtype)

        input_size_h = input.size(-2)
        input_size_w = input.size(-1)

        grid = lambda meta: (triton.cdiv(num_output_features, meta["BLOCK_H"]), triton.cdiv(num_output_features, meta["BLOCK_W"]))

        jitted_kernel = JITFunction(adaptive_avg_pool2d_kernel[grid])
        for pid_n in range(num_batches):
            for pid_c in range(num_output_features):
                jitted_kernel[(pid_n, pid_c)](
                    input,
                    output,
                    output_size_h,
                    output_size_w,
                    input_size_h,
                    input_size_w,
                    num_output_features,
                )

        ctx.save_for_backward(input)
        ctx.output_size = output_size

        return output


def adaptive_avg_pool2d(input, output_size):
    return AdaptiveAvgPool2d.apply(input, output_size)
