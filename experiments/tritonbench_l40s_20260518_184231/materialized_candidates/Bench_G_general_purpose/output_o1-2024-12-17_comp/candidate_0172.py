import torch
import triton
import triton.language as tl
from triton.autotune import Config, heuristics

@triton.autotune(
    configs=[
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4}),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4}),
        Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'num_warps': 1}),
    ],
    key=['M', 'N']
)
@heuristics({
    'num_warps': lambda args: 4 if args['N'] >= 128 else 1
})
@triton.jit
def log_softmax_kernel(
    x_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    **metaparams
):
    BLOCK_M = metaparams['BLOCK_M']
    BLOCK_N = metaparams['BLOCK_N']
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N
    x_offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
    x = tl.where(mask_m[:, None] & mask_n[None, :], tl.load(x_ptr + x_offsets, mask=[mask_m[:, None] & mask_n[None, :]]), float('-inf'))
    max_val = tl.max(x, 1)
    x = x - max_val[:, None]
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, 1)
    log_sum_exp_x = tl.log(sum_exp_x)
    log_softmax_val = x - log_sum_exp_x[:, None]
    out_offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
    tl.store(output_ptr + out_offsets, log_softmax_val, mask=[mask_m[:, None] & mask_n[None, :]])

@triton.autotune(
    configs=[
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4}),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4}),
        Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'num_warps': 1}),
    ],
    key=['M', 'N']
)
@heuristics({
    'num_warps': lambda args: 4 if args['N'] >= 128 else 1
})
@triton.jit
def log_softmax_backward_kernel(
    grad_output_ptr, output_ptr, grad_input_ptr,
    M, N,
    stride_m, stride_n,
    **metaparams
):
    BLOCK_M = metaparams['BLOCK_M']
    BLOCK_N = metaparams['BLOCK_N']
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N
    go_offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
    out_offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
    grad_output = tl.where(mask_m[:, None] & mask_n[None, :],
                           tl.load(grad_output_ptr + go_offsets, mask=[mask_m[:, None] & mask_n[None, :]]),
                           0.0)
    output = tl.where(mask_m[:, None] & mask_n[None, :],
                      tl.load(output_ptr + out_offsets, mask=[mask_m[:, None] & mask_n[None, :]]),
                      0.0)
    sum_grad = tl.sum(grad_output, 1)
    grad_input_val = grad_output - tl.exp(output) * sum_grad[:, None]
    gi_offsets = rm[:, None] * stride_m + rn[None, :] * stride_n
    tl.store(grad_input_ptr + gi_offsets, grad_input_val, mask=[mask_m[:, None] & mask_n[None, :]])

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        dim_size = x.size(dim)
        x_reshaped = x.transpose(dim, -1).contiguous()
        M = x_reshaped.numel() // dim_size
        N = dim_size
        out = torch.empty_like(x_reshaped)
        grid = lambda meta: ( (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'], )
        log_softmax_kernel[grid](x_reshaped, out, M, N, out.stride(0), out.stride(1))
        ctx.save_for_backward(out, torch.tensor(dim))
        return out.transpose(dim, -1)

    @staticmethod
    def backward(ctx, grad_output):
        out, dim_t = ctx.saved_tensors
        dim = dim_t.item()
        dim_size = out.size(-1)
        grad_output_reshaped = grad_output.transpose(dim, -1).contiguous()
        M = grad_output_reshaped.numel() // dim_size
        N = dim_size
        grad_input = torch.empty_like(grad_output_reshaped)
        grid = lambda meta: ( (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'], )
        log_softmax_backward_kernel[grid](
            grad_output_reshaped, out, grad_input, M, N,
            grad_input.stride(0), grad_input.stride(1)
        )
        return grad_input.transpose(dim, -1), None

def log_softmax(x, dim=-1):
    return LogSoftmax.apply(x, dim)
