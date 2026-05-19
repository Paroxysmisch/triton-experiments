import triton
import triton.language as tl
import logging

@triton.jit
def exp_mean_kernel(x, y, x_arg, y_arg, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Get program ids for x and y blocks
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # Calculate row and column offsets
    rows_offset_x = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset_x = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    rows_offset_y = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    
    # Compute offsets for x and y
    offset_x = rows_offset_x * N + cols_offset_x
    offset_y = rows_offset_y * N + cols_offset_y
    
    # Create masks for valid indices
    mask_cols = cols_offset_x < N
    mask_x = (rows_offset_x < M) & mask_cols
    mask_y = (rows_offset_y < M) & mask_cols
    
    # Load x values and compute mean
    x_values = tl.load(x + offset_x, mask=mask_x, other=0.)
    x_exp = tl.exp(x_values)
    x_sum = tl.sum(x_exp, axis=1)
    
    # Load y values and compute weighted sum
    y_values = tl.load(y + offset_y, mask=mask_y, other=0.)
    y_weighted = y_values * x_sum
    
    # Store result
    tl.store(y_arg + offset_y, y_weighted, mask=mask_y)

logging.debug("GEMS EXP MEAN")
exp_mean = lambda x, dim=None, keepdim=False, dtype=None, out=None: \
    _reduce_local(x, None, None, None, None, None, None, None, None, None, None, None, None, None, 
                  reduce_op='EXP_MEAN', dtype=dtype, keepdim=keepdim, dim=dim, out=out)
