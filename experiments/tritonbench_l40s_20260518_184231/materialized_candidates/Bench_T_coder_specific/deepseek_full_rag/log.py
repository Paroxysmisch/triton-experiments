import triton
import triton.language as tl

@triton.jit
def log_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.log(x)
    tl.store(output_ptr + offsets, result, mask=mask)

def log(input, *, out=None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not all(isinstance(x, (int, float)) for x in input.flatten().tolist()):
        raise ValueError("input must be a floating-point tensor")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("shape of out must be the same as input")
    else:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    log_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out

# Test the log function
x = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
out = log(x)
print(f"Input: {x}")
print(f"Output: {out}")
