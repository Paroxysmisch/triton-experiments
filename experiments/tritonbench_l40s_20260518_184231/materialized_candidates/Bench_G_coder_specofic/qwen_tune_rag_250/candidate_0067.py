import torch
import triton
import triton.language as tl
from torch.autograd.function import Function

class RmsNorm(Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-6):
        # save for backward
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        x_shape = x.shape
        # flatten x to 2D tensor, N is the number of elements in the last dimension
        x = x.view(-1, x.shape[-1])
        B, N = x.shape
        M = x.shape[-1]
        # stride of x, stride_x_batch between batches, stride_x_m between elements in a row, stride_x_k between elements in a column
        stride_x_batch, stride_x_m, stride_x_k = x.stride()
        # reshape x for triton
        x = x.contiguous().view(B * N,)
        out = torch.empty_like(x)
        # RMSNorm kernel
        def rms_norm_kernel(pid, x_ptr, rms_w_ptr, out_ptr,
                            stride_x_batch, stride_x_m, stride_x_k,
                            stride_rms_w,
                            stride_out_batch, stride_out_m, stride_out_k,
                            N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_M_SIZE: tl.constexpr):
            # parallel at m dimension
            offset_m = pid * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
            m_mask = offset_m[:, None] < M
            x_ptr_mask = offset_m < M
            # load
            rms_w_offset = tl.load(rms_w_ptr + offset_m * stride_rms_w, mask=m_mask, other=0)
            x = tl.load(x_ptr + offset_m * stride_x_k, mask=x_ptr_mask, other=0).to(tl.float32)
            # compute
            mean = tl.sum(x, axis=0) / N
            x = x - mean
            x_bar = tl.where(x_ptr_mask, x, 0.)
            var = tl.sum(x_bar * x_bar, axis=0) / N
            rrms = 1. / tl.sqrt(var + eps)
            # write-back
            out = (x * rrms).to(Y.dtype.element_ty) * rms_w_offset
            tl.store(out_ptr + offset_m * stride_out_k, out, mask=m_mask)
        # launch kernel
        rms_norm_kernel[(B,)](rms_norm_kernel,
                               x, weight, out,
                               stride_x_batch, stride_x_m, stride_x_k,
                               *weight.stride(),
                               *out.stride(),
                               N_SIZE=N, eps=eps, BLOCK_M_SIZE=1024,
                               num_warps=8)
        # restore shape
        out = out.view(x_shape)
        return out

def rms_norm(x, normalized_shape, weight, eps=1e-6):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
