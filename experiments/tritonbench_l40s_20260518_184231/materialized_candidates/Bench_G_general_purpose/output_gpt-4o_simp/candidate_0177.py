import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 32, 'num_warps': 1}),
        triton.Config({'BT': 64, 'num_warps': 2}),
        triton.Config({'BT': 128, 'num_warps': 4}),
    ],
    key=['N']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(s_ptr, out_ptr, N, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Offsets for loading the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    mask = offsets < N

    # Load data from the input tensor
    s = tl.load(s_ptr + offsets, mask=mask, other=0.0)

    # Create a lower triangular mask for cumulative sum computation
    lt_mask = tl.arange(0, BLOCK_SIZE)[:, None] <= tl.arange(0, BLOCK_SIZE)[None, :]

    # Compute the cumulative sum using matrix multiplication with the lower triangular mask
    cumsum = tl.dot(lt_mask.to(tl.float32), s.to(tl.float32))

    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, cumsum, mask=mask)


def chunk_global_cumsum_vector(s, out, BLOCK_SIZE=64):
    assert s.ndim == 2, "Input tensor must be 2D"
    N, stride = s.shape

    # Launch the kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    chunk_global_cumsum_vector_kernel[grid](
        s, out, N, stride, BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
import torch

# Create a sample 2D tensor
s = torch.rand((128, 128), dtype=torch.float32).cuda()
out = torch.empty_like(s)

# Compute the chunk-based cumulative sum
chunk_global_cumsum_vector(s, out)

# Print the result
print(out)
