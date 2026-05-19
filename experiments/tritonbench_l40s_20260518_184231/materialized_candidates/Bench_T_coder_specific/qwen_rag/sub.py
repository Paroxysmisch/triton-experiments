import triton
import triton.language as tl

@triton.jit
def sub_kernel(
    X_ptr, Y_ptr, Z_ptr,
    x_numel, y_numel, z_numel,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_Z: tl.constexpr
):
    pid = tl.program_id(axis=(0, 1, 2))
    x_block_start = pid[0] * BLOCK_SIZE_X
    y_block_start = pid[1] * BLOCK_SIZE_Y
    z_block_start = pid[2] * BLOCK_SIZE_Z
    
    x_offsets = x_block_start + tl.arange(0, BLOCK_SIZE_X)
    y_offsets = y_block_start + tl.arange(0, BLOCK_SIZE_Y)
    z_offsets = z_block_start + tl.arange(0, BLOCK_SIZE_Z)
    
    x_mask = x_offsets < x_numel
    y_mask = y_offsets < y_numel
    z_mask = z_offsets < z_numel
    
    x_val = tl.load(X_ptr + x_offsets, mask=x_mask, other=0)
    y_val = tl.load(Y_ptr + y_offsets, mask=y_mask, other=0)
    z_val = x_val - alpha * y_val
    
    tl.store(Z_ptr + z_offsets, z_val, mask=z_mask)
