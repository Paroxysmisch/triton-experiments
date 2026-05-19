import torch
import triton
import triton.language as tl

@triton.jit
def sum_kernel(input, output, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of input it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    input = input + pid * N
    row_mask = pid < M

    # Compute sum
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(input + cols, mask, other=0.0).to(tl.float32)
        _sum += a
    total_sum = tl.sum(_sum, axis=1)
    tl.store(output, total_sum, row_mask)

def sum(input: torch.Tensor, dim, keepdim=False, *, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = input.dtype
    if dim is None:
        return input.sum(dtype=dtype)

    shape = list(input.shape)
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % input.ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = input.numel() // N
    output = torch.empty(shape, dtype=dtype, device=input.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch.cuda.device(input.device):
        sum_kernel[grid](input, output, M, N, BLOCK_M=8, BLOCK_N=8)
    
    if not keepdim:
        output = output.squeeze(dim)
    return output

# Example usage
b = torch.randn(2, 3, 4, 5, device="cuda")
print(sum(b, [1, 2]))
