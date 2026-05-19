import triton
import triton.language as tl

@triton.jit
def affine_grid_kernel(theta_ptr, grid_ptr, N, C, H_out, W_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid // W_out
    col = pid % W_out
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    theta = tl.load(theta_ptr + pid * 6, mask=mask)
    
    for i in range(BLOCK_SIZE):
        h = offsets[i]
        grid_idx = h * H_out * W_out + row * W_out + col
        x = float(col) / (W_out - 1) * 2.0 - 1.0
        y = float(row) / (H_out - 1) * 2.0 - 1.0
        
        grid_x = theta[0] * x + theta[1] * y + theta[2]
        grid_y = theta[3] * x + theta[4] * y + theta[5]
        
        tl.store(grid_ptr + grid_idx, (grid_x, grid_y), mask=mask)
