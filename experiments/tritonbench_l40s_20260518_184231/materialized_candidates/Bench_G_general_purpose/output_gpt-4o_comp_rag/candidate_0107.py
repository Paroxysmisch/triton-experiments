import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, input_row_stride, output_row_stride, M, N, TILE_N: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, TILE_N)
    
    # Initialize pointers to the start of the current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Initialize accumulators for the maximum and sum of exponentials
    max_acc = tl.zeros((TILE_N,), dtype=tl.float32) - float('inf')
    exp_sum_acc = tl.zeros((TILE_N,), dtype=tl.float32)

    # Phase 1: Reduction to compute max and sum of exponentials
    for i in range(0, N, TILE_N):
        input_ptrs = row_start_ptr + i + col_offsets
        mask = col_offsets < (N - i)
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        max_acc = tl.max(tl.where(mask, row, max_acc), max_acc)
        row_minus_max = row - max_acc
        exp_vals = tl.exp(row_minus_max)
        exp_sum_acc += tl.where(mask, exp_vals, 0.0)
    
    # Phase 2: Compute the final softmax values and store them in the output
    for i in range(0, N, TILE_N):
        input_ptrs = row_start_ptr + i + col_offsets
        output_ptrs = output_row_start_ptr + i + col_offsets
        mask = col_offsets < (N - i)
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        row_minus_max = row - max_acc
        exp_vals = tl.exp(row_minus_max)
        softmax_output = exp_vals / exp_sum_acc
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x, TILE_N=128):
    M, N = x.shape
    y = torch.empty_like(x)
    
    # Determine the number of warps based on TILE_N
    num_warps = 4
    if TILE_N >= 2048:
        num_warps = 8
    if TILE_N >= 4096:
        num_warps = 16

    # Enqueue the kernel
    softmax_kernel_online_v2[(M,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        M,
        N,
        TILE_N,
        num_warps=num_warps
    )
    
    return y

# Example usage
x = torch.randn(128, 512, device='cuda')
y = softmax(x)
print(y)
