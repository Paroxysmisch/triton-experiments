import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    a_row_stride, a_col_stride,
    b_row_stride, b_col_stride,
    c_row_stride, c_col_stride,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    NUM_SMS
):
    pid = tl.program_id(axis=0)
    sm_id = pid // NUM_SMS

    # load the current block of A from global memory
    a = tl.load(a_ptr + pid * BLOCK_SIZE_M * a_row_stride + sm_id * BLOCK_SIZE_K * a_col_stride)

    # load the current block of B from global memory
    b = tl.load(b_ptr + pid * BLOCK_SIZE_N * b_row_stride + sm_id * BLOCK_SIZE_K * b_col_stride)

    # initialize the accumulator with zero
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # perform the matrix multiplication
    for k in range(K):
        accumulator += tl.dot(a[k], b[k])

    # load the current block of C from global memory
    c = tl.load(c_ptr + pid * BLOCK_SIZE_M * c_row_stride + sm_id * BLOCK_SIZE_N * c_col_stride)

    # combine the loaded C with the computed accumulator
    c += accumulator

    # store the updated block of C back to global memory
    tl.store(c_ptr + pid * BLOCK_SIZE_M * c_row_stride + sm_id * BLOCK_SIZE_N * c_col_stride, c)

# Wrapper function
def matmul_persistent(a, b, c, grid=None, num_sms=1):
    assert a.shape[1] == b.shape[0]
    assert a.shape[0] == c.shape[0]
    assert b.shape[1] == c.shape[1]

    block_size = min(a.shape[1], num_sms)

    if grid is None:
        grid = (min(a.numel() // block_size, 65536), 1, 1)

    a_ptr = a.flatten().data_ptr()
    b_ptr = b.flatten().data_ptr()
    c_ptr = c.flatten().data_ptr()

    a_row_stride = tl.shape(a, grouped=True)[0]
    a_col_stride = tl.shape(a, grouped=True)[1]
    b_row_stride = tl.shape(b, grouped=True)[0]
    b_col_stride = tl.shape(b, grouped=True)[1]
    c_row_stride = tl.shape(c, grouped=True)[0]
    c_col_stride = tl.shape(c, grouped=True)[1]

    M, N, K = a.shape[0], b.shape[1], a.shape[1]

    matmul_kernel_persistent[grid](
        a_ptr, b_ptr, c_ptr, 
        M, N, K, 
        a_row_stride, a_col_stride,
        b_row_stride, b_col_stride,
        c_row_stride, c_col_stride,
        block_size, block_size, block_size,
        num_sms
    )
