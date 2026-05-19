import torch
import triton
import triton.language as tl

@triton.jit
def sum_dim_kernel(X, Sum, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask_rows = rows < M

    X_ptr = X + rows[:, None] * N
    sum_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask_cols = cols < N
        mask = mask_rows[:, None] & mask_cols[None, :]
        a = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc += a

    total_sum = tl.sum(sum_acc, axis=1)
    Sum_ptr = Sum + rows
    tl.store(Sum_ptr, total_sum.to(Sum.dtype.element_ty), mask=mask_rows)

def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

def sum(input, dim, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    if dim is None:
        out = torch.sum(input, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * input.ndim)
        return out
    
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % input.ndim for d in dim]
    x_compressed = dim_compress(input, dim)
    
    original_shape = list(input.shape)
    N = 1
    for d in dim:
        N *= original_shape[d]
    M = x_compressed.numel() // N
    
    output_shape = list(original_shape)
    for d in dim:
        output_shape[d] = 1
    out = torch.empty(output_shape, dtype=dtype, device=input.device)
    
    BLOCK_M = 128
    BLOCK_N = 64
    grid = lambda META: (triton.cdiv(M, BLOCK_M),)
    
    sum_dim_kernel[grid](x_compressed, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    if not keepdim:
        out = out.squeeze(dim)
    return out

# Test case
b = torch.randn(2, 3, 4, 5, device="cuda")
print(sum(b, [1, 2]))
