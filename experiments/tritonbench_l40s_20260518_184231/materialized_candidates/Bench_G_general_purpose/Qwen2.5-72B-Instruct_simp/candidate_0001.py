import triton
import triton.language as tl

@triton.jit
def _dequantize_rowwise(x_ptr, state_x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, M)

    for row in range(row_start, row_end):
        max_val = tl.load(state_x_ptr + row)
        inv_127 = 1.0 / 127.0
        row_offset = row * N
        for col in range(N):
            x_val = tl.load(x_ptr + row_offset + col)
            output_val = x_val * max_val * inv_127
            tl.store(output_ptr + row_offset + col, output_val)

import torch
import triton
import triton.language as tl

def dequantize_rowwise(x, state_x):
    M, N = x.shape
    output = torch.empty_like(x, dtype=torch.float32, device=x.device)
    
    # Calculate the nearest power of two for the number of columns
    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    _dequantize_rowwise[grid](x, state_x, output, M, N, BLOCK_SIZE)
    
    return output

import torch

# Example input tensor and state tensor
x = torch.randint(0, 128, (1024, 512), device='cuda')
state_x = torch.rand(1024, device='cuda')

# Perform row-wise dequantization
output = dequantize_rowwise(x, state_x)

print(output)
