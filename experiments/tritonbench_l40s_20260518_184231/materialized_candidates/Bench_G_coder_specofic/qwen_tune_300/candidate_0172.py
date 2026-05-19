import torch
import triton
import triton.language as tl
from triton import autotune, heuristics

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=32),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=32),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=16),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=32),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=16),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=32),
    ],
    key=["M", "N"],
    prune_configs_by={
        "early_config_prune": heuristics.autotune_prune_configs_by_dim_size,
    },
)
@triton.jit
def log_softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Map the program id to the row of the log softmax it should compute.
    row_start = tl.program_id(0) * BLOCK_M
    row_step = tl.num_programs(0) * BLOCK_M
    row_end = M
    for row_idx in tl.range(row_start, row_end, row_step, BLOCK_M):
        row = row_idx // BLOCK_M
        col_offsets = tl.arange(0, BLOCK_N)
        row_mask = col_offsets < N

        input_ptr_mask = input_ptr + row * input_row_stride + col_offsets
        row_input = tl.load(input_ptr_mask, mask=row_mask, other=-float("inf")).to(tl.float32)

        row_max = tl.max(row_input, axis=0)
        row_input_minus_max = row_input - row_max
        numerator = tl.exp(row_input_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = row_input_minus_max - tl.log(denominator)

        output_row_start = row * output_row_stride
        output_ptr_mask = output_ptr + output_row_start + col_offsets
        tl.store(output_ptr_mask, softmax_output, mask=row_mask)

@triton.jit
def log_softmax_backward_kernel(
    output_grad_ptr, input_output_ptr, input_ptr, input_row_stride, output_row_stride, M, N
):
    # Map the program id to the row of the log softmax it should compute.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    row_end = M
    for row_idx in tl.range(row_start, row_end, row_step, 1):
        row = row_idx
        col_offsets = tl.arange(0, N)
        row_mask = col_offsets < N

        input_ptr_mask = input_ptr + row * input_row_stride + col_offsets
        row_input = tl.load(input_ptr_mask, mask=row_mask, other=-float("inf")).to(tl.float32)

        input_output_ptr_mask = input_output_ptr + row * output_row_stride + col_offsets
        row_input_output = tl.load(input_output_ptr_mask, mask=row_mask, other=-float("inf")).to(tl.float32)

        output_grad_ptr_mask = output_grad_ptr + row * output_row_stride + col_offsets
        row_output_grad = tl.load(output_grad_ptr_mask, mask=row_mask, other=0).to(tl.float32)

        input_minus_output = row_input - row_input_output
        softmax_grad = row_output_grad - tl.exp(input_minus_output) * tl.sum(row_output_grad, axis=0)

        tl.store(output_grad_ptr_mask, softmax_grad, mask=row_mask)

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        if dim >= x.ndim:
            raise ValueError("dim must be less than x.ndim")
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float16)

        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]

        x = x.contiguous()

        output = torch.empty_like(x)
        input = dim_permute(x, dim, 0)
        input_stride = input.stride()
        output_stride = output.stride()
        input = dim_compress(input, 0)
        output = dim_compress(output, 0)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        log_softmax_kernel[grid](output, input, input_stride[0], output_stride[0], M, N)

        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, output_grad):
        dim = ctx.dim
        (output,) = ctx.saved_tensors

        output_grad = output_grad.contiguous()
        input_output = output.contiguous()
        input = dim_permute(output, 0, dim)
        input_stride = input.stride()
        output_grad_stride = output_grad.stride()
        input = dim_compress(input, 0)
        output_grad = dim_compress(output_grad, 0)

        grid = lambda meta: (triton.cdiv(input.shape[0], meta["BLOCK_M"]),)
        log_softmax_backward_kernel[grid](
            output_grad, input_output, input, input_stride[0], output_grad_stride[0], input.shape[0], input.shape[1]
        )

        input_grad = dim_permute(input, 0, dim)
        input_grad = dim_expand(input_grad, list(range(len(input_grad.shape) - 1)), output_grad.shape)
        return input_grad, None

def log_softmax(x, dim=-1, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    return LogSoftmax.apply(x, dim)
