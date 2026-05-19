import triton
import triton.language as tl
import torch

@triton.jit
def _logsumexp_kernel(
    X, OUT, x_row_stride, x_col_stride, out_row_stride,
    N_COLS, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = X + row_idx * x_row_stride
    max_value = tl.float32(-float('inf'))
    sum_exp = tl.float32(0.0)
    
    for block_col in range(0, N_COLS, BLOCK_SIZE):
        col_offsets = block_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_COLS
        x_ptrs = row_start_ptr + col_offsets * x_col_stride
        x = tl.load(x_ptrs, mask=mask, other=-tl.float32(float('inf')))
        
        curr_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_value, curr_max)
        
        sum_exp *= tl.exp(max_value - new_max)
        sum_exp += tl.sum(tl.exp(x - new_max), axis=0)
        max_value = new_max
    
    out = tl.log(sum_exp) + max_value
    out_ptr = OUT + row_idx * out_row_stride
    tl.store(out_ptr, out)

def logsumexp(input: torch.Tensor, dim: int, keepdim: bool = False, *, out: torch.Tensor = None) -> torch.Tensor:
    if not input.is_cuda:
        return torch.logsumexp(input, dim, keepdim=keepdim, out=out)
    
    dim = dim if dim >= 0 else input.dim() + dim
    input_tr = input.transpose(dim, -1)
    *leading_dims, reduction_size = input_tr.shape
    input_2d = input_tr.contiguous().view(-1, reduction_size)
    M, N = input_2d.shape
    
    output_1d = input_2d.new_empty(M)
    BLOCK_SIZE = 1024
    grid = (M,)
    _logsumexp_kernel[grid](
        input_2d, output_1d, input_2d.stride(0), input_2d.stride(1), output_1d.stride(0),
        N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    output = output_1d.view(*leading_dims)
    if keepdim:
        output = output.unsqueeze(-1).transpose(dim, -1)
    
    if out is not None:
        out.copy_(output)
        return out
    return output
