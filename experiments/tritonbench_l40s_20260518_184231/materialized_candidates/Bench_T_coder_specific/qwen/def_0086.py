import triton
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=8),
    ],
    key=['n_elements']
)
def log_tanh(input, out=None, BLOCK_SIZE=256):
    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    log_tanh_kernel[grid](input.contiguous(), out.contiguous(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage
input_tensor = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32, device='cuda')
output_tensor = log_tanh(input_tensor)
print(output_tensor)
