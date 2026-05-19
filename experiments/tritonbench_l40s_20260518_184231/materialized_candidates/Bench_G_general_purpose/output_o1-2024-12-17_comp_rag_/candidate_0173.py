import torch
import triton
import triton.language as tl


# ----------------------------------------------------------
#  log_softmax_kernel
# ----------------------------------------------------------
@triton.heuristics({
    # Suggest block sizes based on the problem size
    'BLOCK_M': lambda *args, **meta: 128,
    'BLOCK_N': lambda *args, **meta: 128,
})
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def log_softmax_kernel(
    x_ptr, 
    output_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    # Create block of indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Create pointers for reading input
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_n[None, :] * stride_xn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Load input block
    x = tl.where(mask, tl.load(x_ptrs, mask=mask), float("-inf"))
    # Row-wise max
    row_max = tl.max(x, 1)
    x = x - row_max[:, None]
    # Exponentials and sum
    exp_x = tl.exp(x)
    denominator = tl.sum(exp_x, 1)
    # Normalized log softmax
    log_softmax_val = x - tl.log(denominator)[:, None]
    # Write output
    y_ptrs = output_ptr + (offs_m[:, None] * stride_ym) + (offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, log_softmax_val, mask=mask)


# ----------------------------------------------------------
#  log_softmax_backward_kernel
# ----------------------------------------------------------
@triton.heuristics({
    'BLOCK_M': lambda *args, **meta: 128,
    'BLOCK_N': lambda *args, **meta: 128,
})
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4}, num_stages=2),
        triton.Config({'num_warps': 8}, num_stages=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def log_softmax_backward_kernel(
    grad_out_ptr,
    output_ptr,
    grad_in_ptr,
    M, N,
    stride_gom, stride_gon,
    stride_om, stride_on,
    stride_gim, stride_gin,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    # Create block of indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Pointers
    grad_out_ptrs = grad_out_ptr + (offs_m[:, None] * stride_gom) + (offs_n[None, :] * stride_gon)
    out_ptrs = output_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Load
    grad_out = tl.where(mask, tl.load(grad_out_ptrs), 0.0)
    out_val = tl.where(mask, tl.load(out_ptrs), 0.0)
    # Sum of grad along the row
    grad_sum = tl.sum(grad_out, 1)
    # Compute grad input
    # log_softmax_backward: grad_in = grad_out - exp(output) * sum(grad_out)
    # but exp(output) = the original softmax.
    softmax_val = tl.exp(out_val)
    grad_in_block = grad_out - softmax_val * grad_sum[:, None]
    # Write
    grad_in_ptrs = grad_in_ptr + (offs_m[:, None] * stride_gim) + (offs_n[None, :] * stride_gin)
    tl.store(grad_in_ptrs, grad_in_block, mask=mask)


# ----------------------------------------------------------
#  LogSoftmax PyTorch Autograd
# ----------------------------------------------------------
class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        x_contig = x.contiguous()
        # Ensure dimension is positive
        if dim < 0:
            dim = x_contig.ndim + dim
        # Move dim to the last axis for convenience
        perm = list(range(x_contig.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_transposed = x_contig.permute(perm)
        M, N = x_transposed[:-1].numel(), x_transposed.size(-1)

        # Allocate output
        out_transposed = torch.empty_like(x_transposed)

        # Launch forward kernel
        grid = ( (M + 127) // 128, )
        log_softmax_kernel[grid](
            x_ptr=x_transposed, 
            output_ptr=out_transposed,
            M=M, 
            N=N,
            stride_xm=x_transposed.stride(0),
            stride_xn=x_transposed.stride(-1),
            stride_ym=out_transposed.stride(0),
            stride_yn=out_transposed.stride(-1)
        )
        # Reshape back
        out = out_transposed.permute(perm)
        ctx.save_for_backward(out, torch.tensor(dim, device=x.device))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, dim_t = ctx.saved_tensors
        dim = int(dim_t.item())
        grad_output_contig = grad_output.contiguous()
        # Move dim to last axis
        perm = list(range(grad_output_contig.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        grad_out_transposed = grad_output_contig.permute(perm)
        out_transposed = out.permute(perm)

        M, N = out_transposed[:-1].numel(), out_transposed.size(-1)
        grad_in_transposed = torch.empty_like(out_transposed)

        # Launch backward kernel
        grid = ( (M + 127) // 128, )
        log_softmax_backward_kernel[grid](
            grad_out_ptr=grad_out_transposed,
            output_ptr=out_transposed,
            grad_in_ptr=grad_in_transposed,
            M=M, 
            N=N,
            stride_gom=grad_out_transposed.stride(0),
            stride_gon=grad_out_transposed.stride(-1),
            stride_om=out_transposed.stride(0),
            stride_on=out_transposed.stride(-1),
            stride_gim=grad_in_transposed.stride(0),
            stride_gin=grad_in_transposed.stride(-1),
        )

        # Reshape back
        grad_in = grad_in_transposed.permute(perm)
        return grad_in, None


# ----------------------------------------------------------
#  log_softmax wrapper
# ----------------------------------------------------------
def log_softmax(x, dim=-1):
    return LogSoftmax.apply(x, dim)
