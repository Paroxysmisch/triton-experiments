import triton
import triton.language as tl

@triton.jit
def log1p_kernel(input_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.log1p(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.double)
    else:
        assert out.shape == input.shape
        assert out.stride() == input.stride()
        assert out.is_contiguous()
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
result_torch = torch.log(1 + x)
result_triton = log1p(x)
assert torch.allclose(result_torch, result_triton.to(torch.double), atol=1e-8)
