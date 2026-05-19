import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 2}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 2}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 2}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 2}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 2}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4}, num_stages=3, num_warps=8),
    ],
    key=["C", "feature_dim"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args: configs[named_args["feature_dim"]],
        "perf_model": None,
    },
)
@triton.heuristics({
    "EVEN_M": lambda args: args["feature_dim"] % 128 == 0,
})
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    lse,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    m_range = offs_m + BLOCK_M if EVEN_M else BLOCK_M
    m = tl.arange(0, m_range)
    n = tl.arange(0, BLOCK_N)
    offsets = m[:, None] * N + n[None, :]
    input_ptrs = input_ptr + offsets
    mask = m[:, None] < M

    log_softmax_values = tl.load(input_ptrs, mask=mask, other=float("-inf")).to(tl.float32)
    log_softmax_values = log_softmax_values - tl.max(log_softmax_values, axis=1)[:, None]
    lse_values = tl.log(tl.sum(tl.exp(log_softmax_values), axis=1))
    log_softmax_values = log_softmax_values - lse_values[:, None]

    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, log_softmax_values, mask=mask)
    tl.store(lse_ptr + m, lse_values)


@triton.jit
def log_softmax_backward_kernel(
    output_ptr,
    log_softmax_ptr,
    M,
    N,
    lse_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    m_range = offs_m + BLOCK_M if EVEN_M else BLOCK_M
    m = tl.arange(0, m_range)
    n = tl.arange(0, BLOCK_N)
    offsets = m[:, None] * N + n[None, :]
    log_softmax_ptrs = log_softmax_ptr + offsets
    mask = m[:, None] < M

    log_softmax_values = tl.load(log_softmax_ptrs, mask=mask, other=float("-inf")).to(tl.float32)
    lse = tl.load(lse_ptr + m)
    output_values = log_softmax_values - lse[:, None]

    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, output_values, mask=mask)


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, dtype):
        assert dtype in (None, "float16", "bfloat16", "float32", "float64")
        input_arg = input.contiguous()
        if dtype is None:
            dtype = str(input.dtype).split(".")[1]
        out_dtype = getattr(torch, dtype)
        out = torch.empty_like(input, dtype=out_dtype, memory_format=torch.contiguous_format)
        input_ = input.to(torch.float32)
        M = int(input.numel(dim=dim) / input.size(dim=dim))
        N = input.size(dim=dim)
        dim = input.ndim - 1 - dim if dim < 0 else dim
        dim_size = input.size(dim=dim)
        input_.movedim(dim, -1)

        grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"]))
        with torch.cuda.device(input.device):
            lse = torch.empty(M, device=input.device)
            log_softmax_kernel[grid](
                out,
                input_,
                M,
                N,
                lse,
                feature_dim=dim_size,
                num_warps=2,
            )
        ctx.save_for_backward(out)
        out = out.movedim(-1, dim)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        dim = ctx.dim
        out_grad_arg = out_grad.contiguous()
        (out,) = ctx.saved_tensors
        M = int(out.numel(dim=dim) / out.size(dim=dim))
        N = out.size(dim=dim)
        dim = out.ndim - 1 - dim if dim < 0 else dim
        dim_size = out.size(dim=dim)
        out.movedim(dim, -1)
        out_grad.movedim(dim, -1)

        input_ = torch.empty_like(out)
        grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"]))
        with torch.cuda.device(out.device):
            log_softmax_backward_kernel[grid](
                input_,
                out,
                M,
                N,
                lse=lse,
                feature_dim=dim_size,
                num_warps=2,
            )
        input_ = input_.movedim(-1, dim)
        out_grad = out_grad.movedim(-1, dim)
        return input_ * out_grad, None, None


def log_softmax(input, dim=-1, dtype=None):
