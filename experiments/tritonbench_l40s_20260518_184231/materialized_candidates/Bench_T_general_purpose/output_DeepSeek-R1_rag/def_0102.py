import triton
import triton.language as tl
import torch

# Use the existing logsumexp function from Document 1
@triton.jit
def _logsumexp(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(res) + alpha
    rm = tl.program_id(0) + tl.arange(0, 1)
    OUT = OUT + rm * out_stride
    tl.store(OUT, out)

def logsumexp(input):
    assert input.is_cuda
    *dims, N = input.shape
    input = input.view(-1, N)
    out = input.new_empty(*dims).view(-1)
    M = input.shape[0]
    _logsumexp[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                     BLOCK_N=4096, num_warps=4)
    return out.view(*dims)

@triton.jit
def _softmax_mul_kernel(
    X, L, O, OUT,
    x_row_stride, x_col_stride,
    l_row_stride,
    o_row_stride, o_col_stride,
    out_row_stride, out_col_stride,
    N_COLS,
    IS_SCALAR_O: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < N_COLS

    # Load logsumexp for this row
    l_row_ptr = L + row_idx * l_row_stride
    logsumexp_val = tl.load(l_row_ptr)

    # X pointer for this row
    x_row_ptr = X + row_idx * x_row_stride
    # O pointer for this row (if not scalar)
    if not IS_SCALAR_O:
        o_row_ptr = O + row_idx * o_row_stride
    # OUT pointer for this row
    out_row_ptr = OUT + row_idx * out_row_stride

    for col_block in range(0, N_COLS, BLOCK_SIZE):
        col_idx = col_block + col_offsets
        x_ptr = x_row_ptr + col_idx * x_col_stride
        x_val = tl.load(x_ptr, mask=col_mask, other=float('-inf'))
        softmax_val = tl.exp(x_val - logsumexp_val)
        if IS_SCALAR_O:
            other_val = O
        else:
            o_ptr = o_row_ptr + col_idx * o_col_stride
            other_val = tl.load(o_ptr, mask=col_mask, other=0.0)
        out_val = softmax_val * other_val
        out_ptr = out_row_ptr + col_idx * out_col_stride
        tl.store(out_ptr, out_val, mask=col_mask)

def softmax_mul(input, other, dim, dtype=None, out=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    input = input.contiguous()
    dim = dim if dim >= 0 else dim + input.dim()
    assert 0 <= dim < input.dim()
    
    # Permute to make 'dim' last
    perm = list(range(input.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    input_p = input.permute(perm)
    orig_shape = input_p.shape
    input_2d = input_p.reshape(-1, orig_shape[-1])
    
    logsumexp_vals = logsumexp(input_2d)
    
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    # Broadcast other to input's shape
    other_bc = torch.broadcast_to(other, input.shape)
    other_p = other_bc.permute(perm).reshape(input_2d.shape)
    
    is_scalar = other_bc.numel() == 1
    if is_scalar:
        O = other_p.min().item()  # Get scalar value
    else:
        O = other_p
    
    M, N = input_2d.shape
    BLOCK_SIZE = 128
    grid = (M,)
    out_2d = input_2d.new_empty(input_2d.shape) if out is None else out.permute(perm).reshape_as(input_2d).contiguous()
    
    _softmax_mul_kernel[grid](
        input_2d, logsumexp_vals, O, out_2d,
        input_2d.stride(0), input_2d.stride(1),
        logsumexp_vals.stride(0),
        other_p.stride(0) if not is_scalar else 0,
        other_p.stride(1) if not is_scalar else 0,
        out_2d.stride(0), out_2d.stride(1),
        N,
        IS_SCALAR_O=is_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    out_p = out_2d.reshape(orig_shape).permute(perm)
    if out is not None:
        out.copy_(out_p)
        return out
    return out_p
