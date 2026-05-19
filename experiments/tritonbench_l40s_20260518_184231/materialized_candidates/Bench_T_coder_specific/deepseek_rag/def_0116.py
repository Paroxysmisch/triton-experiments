import torch
import triton
import triton.language as tl

@triton.jit
def sum_kernel(X, Sum, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] 
    X = X + pid * N
    Sum = Sum + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a
    sum = tl.sum(_sum, axis=1)
    sum = sum[:, None]
    tl.store(Sum, sum, row_mask)

def sum_dim(x, dim, keepdim=False, *, dtype=None):
  if dtype is None:
    dtype = x.dtype
  if dim is None:
    out = x.sum(dtype=dtype)
    if not keepdim:
      out = out.reshape([1]*x.ndim)
    return out
  
  shape = list(x.shape)
  if isinstance(dim, int):
     dim = [dim]
  dim = [d % x.ndim for d in dim]
  x = dim_compress(x, dim)
  N = 1
  for i in dim:
    N *= shape[i]
    shape[i] = 1
  M = x.numel() // N
  out = torch.empty(shape, dtype=dtype, device=x.device)
  grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

  with torch.cuda.device(x.device):
    sum_kernel[grid](x, out, M, N, BLOCK_M=8, BLOCK_N=8)
  if not keepdim:
    out = out.squeeze(dim)
  return out

b = torch.randn(2, 3, 4, 5, device="cuda")
print(sum_dim(b, [1, 2]))
