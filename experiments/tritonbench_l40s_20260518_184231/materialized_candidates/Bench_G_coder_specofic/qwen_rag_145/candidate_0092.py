import triton
import triton.language as tl
import torch

@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value):
    # Get unique block id
    block_id = tl.program_id(axis=0)
    # Calculate block-start
    block_start = block_id * BLOCK_SIZE
    # Define offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Acquire a mask
    mask = offsets < N
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask)
    # Perform addition
    output = x + value
    # Sync grid
    tl.sync()
    # Store y to shared output
    tl.store(output_ptr + offsets, output, mask=mask)

def puzzle1(x: torch.Tensor, value=10):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    N = output.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=value)
    return output

def main():
    a = torch.Tensor([4, 5, 3, 2]).to(device=torch.device('cuda'))
    print(puzzle1(a))

if __name__ == '__main__':
    main()
