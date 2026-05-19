import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    Y, X, W,
    stride_y_row, stride_y_col,
    stride_x_row, stride_x_col,
    stride_w,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * stride_x_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load input row
    x_ptrs = X + row_start + col_offsets * stride_x_col
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute variance
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    
    # Load weights
    w_ptrs = W + col_offsets * stride_w
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    
    # Normalize and scale
    y = (x * rrms).to(Y.dtype.element_ty) * w
    
    # Store output
    y_ptrs = Y + pid * stride_y_row + col_offsets * stride_y_col
    tl.store(y_ptrs, y, mask=mask)

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        
        *dims, N = x.shape
        BLOCK_SIZE = triton.next_power_of_two(N)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Ensure minimum block size for efficiency
        
        grid = (torch.prod(torch.tensor(dims)),)
        rms_norm_kernel[grid](
            y, x, weight,
            y.stride(0), y.stride(-1),
            x.stride(0), x.stride(-1),
            weight.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x, weight, torch.tensor(eps))
        return y

    @staticmethod
    def backward(ctx, dy):
        # Backward pass not implemented for brevity
        raise NotImplementedError("Backward pass not implemented")

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    input_shape = x.shape
    x_ = x.reshape(-1, normalized_shape[-1])
    y = RmsNorm.apply(x_, weight, eps)
    return y.reshape(input_shape)
