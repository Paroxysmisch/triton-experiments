import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def mul2_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE_X, BLOCK_SIZE_Y, grid_x, grid_y):
    pid_x, pid_y = tl.program_id(0), tl.program_id(1)
    block_start_x = pid_x * BLOCK_SIZE_X
    block_start_y = pid_y * BLOCK_SIZE_Y
    mask_x = tl.mask(block_start_x + tl.arange(0, BLOCK_SIZE_X) < n_elements)
    mask_y = tl.mask(block_start_y + tl.arange(0, BLOCK_SIZE_Y) < n_elements)
    x = tl.load(x_ptr + block_start_x, mask=mask_x)
    y = tl.load(y_ptr + block_start_y, mask=mask_y)
    tl.store(y_ptr + block_start_y, 2 * x, mask=mask_y)

@triton.jit
def mul2_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE_X, BLOCK_SIZE_Y, grid_x, grid_y):
    pid_x, pid_y = tl.program_id(0), tl.program_id(1)
    block_start_x = pid_x * BLOCK_SIZE_X
    block_start_y = pid_y * BLOCK_SIZE_Y
    mask_x = tl.mask(block_start_x + tl.arange(0, BLOCK_SIZE_X) < n_elements)
    mask_y = tl.mask(block_start_y + tl.arange(0, BLOCK_SIZE_Y) < n_elements)
    x = tl.load(x_ptr + block_start_x, mask=mask_x)
    tl.store(x_ptr + block_start_x, 2 * x, mask=mask_x)

def triton_mul2(x, y):
    n_elements = x.numel()
    grid_x = triton.cdiv(n_elements, BLOCK_SIZE)
    mul2_kernel[grid_x, BLOCK_SIZE](x.data_ptr(), y.data_ptr(), n_elements, BLOCK_SIZE, BLOCK_SIZE, grid_x)

def triton_mul2_inplace(x):
    n_elements = x.numel()
    grid_x = triton.cdiv(n_elements, BLOCK_SIZE)
    mul2_inplace_kernel[grid_x, BLOCK_SIZE](x.data_ptr(), n_elements, BLOCK_SIZE, BLOCK_SIZE, grid_x)
