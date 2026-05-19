import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    a_ptr, v_ptr, alpha_b_ptr, y_ptr,
    n, m,
    stride_am, stride_ak,
    BLOCK_SIZE_K: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n:
        return

    sum = 0.0
    for k in range(0, m, BLOCK_SIZE_K):
        cols = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + row_idx * stride_am + cols * stride_ak
        v_ptrs = v_ptr + cols

        mask = cols < m
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        v = tl.load(v_ptrs, mask=mask, other=0.0)
        sum += tl.sum(a * v)

    s = 1.0 / (1.0 + tl.exp(-sum))
    alpha_b = tl.load(alpha_b_ptr + row_idx)
    y = s - alpha_b

    tl.store(y_ptr + row_idx, y)

class FusedMVSigmoidSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vec, alpha_b):
        n, m = input.shape
        y = torch.empty(n, device=input.device, dtype=input.dtype)
        BLOCK_SIZE_K = 128  # Tune this based on hardware

        grid = (n,)
        fused_mv_sigmoid_sub_kernel[grid](
            input, vec, alpha_b, y,
            n, m,
            input.stride(0), input.stride(1),
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        ctx.save_for_backward(input, vec, alpha_b, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        input, vec, alpha_b, y = ctx.saved_tensors

        s = y + alpha_b
        grad_z = grad_y * s * (1 - s)

        grad_A = torch.outer(grad_z, vec)
        grad_v = torch.mv(input.T, grad_z)
        grad_alpha_b = -grad_y

        return grad_A, grad_v, grad_alpha_b

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    assert input.dim() == 2, "input must be a 2D matrix"
    assert vec.dim() == 1, "vec must be a 1D vector"
    n, m = input.shape
    assert vec.size(0) == m, "vec size does not match input's columns"

    alpha_b = alpha * other
    if not isinstance(alpha_b, torch.Tensor):
        alpha_b = torch.tensor(alpha_b, dtype=input.dtype, device=input.device)
    alpha_b = alpha_b.expand(n).contiguous()

    result = FusedMVSigmoidSub.apply(input, vec, alpha_b)
    if out is not None:
        out.copy_(result)
        return out
    return result
