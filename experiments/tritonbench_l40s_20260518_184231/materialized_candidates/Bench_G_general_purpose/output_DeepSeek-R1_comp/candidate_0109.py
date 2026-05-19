import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    row_start_ptr = input_ptr + row * N
    output_row_ptr = output_ptr + row * N

    # Compute maximum value
    max_val = tl.zeros((1,), dtype=tl.float32) - float('inf')
    aligned_n = tl.math.prev_multiple_of(N, TILE_N)

    # Process aligned tiles for max
    for offs_n in range(0, aligned_n, TILE_N):
        cols = offs_n + tl.arange(0, TILE_N)
        row_data = tl.load(row_start_ptr + cols)
        curr_max = tl.max(row_data, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # Process remaining elements for max
    if aligned_n < N:
        cols = aligned_n + tl.arange(0, TILE_N)
        mask = cols < N
        row_data = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        curr_max = tl.max(row_data, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # Compute sum of exponentials
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    for offs_n in range(0, aligned_n, TILE_N):
        cols = offs_n + tl.arange(0, TILE_N)
        row_data = tl.load(row_start_ptr + cols)
        exp_data = tl.exp(row_data - max_val)
        sum_exp += tl.sum(exp_data, axis=0)

    if aligned_n < N:
        cols = aligned_n + tl.arange(0, TILE_N)
        mask = cols < N
        row_data = tl.load(row_start_ptr + cols, mask=mask, other=0)
        exp_data = tl.exp(row_data - max_val)
        sum_exp += tl.sum(exp_data, axis=0)

    # Compute and store output
    for offs_n in range(0, aligned_n, TILE_N):
        cols = offs_n + tl.arange(0, TILE_N)
        row_data = tl.load(row_start_ptr + cols)
        exp_data = tl.exp(row_data - max_val)
        normalized = exp_data / sum_exp
        tl.store(output_row_ptr + cols, normalized)

    if aligned_n < N:
        cols = aligned_n + tl.arange(0, TILE_N)
        mask = cols < N
        row_data = tl.load(row_start_ptr + cols, mask=mask, other=0)
        exp_data = tl.exp(row_data - max_val)
        normalized = exp_data / sum_exp
        tl.store(output_row_ptr + cols, normalized, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty_like(x)
    TILE_N = 64  # Can be tuned for optimal performance
    assert (TILE_N & (TILE_N - 1)) == 0, "TILE_N must be a power of two"
    grid = (M,)
    softmax_kernel_online_v2[grid](output, x, M, N, TILE_N)
    return output

# Example usage
if __name__ == "__main__":
    x = torch.randn(1000, 1000, device='cuda')
    y = softmax(x)
    print("Input:", x)
    print("Output:", y)
