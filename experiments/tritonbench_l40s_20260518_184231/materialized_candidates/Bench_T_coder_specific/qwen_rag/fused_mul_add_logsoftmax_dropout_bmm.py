import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Determine block size and grid size
    BLOCK_SIZE = 32
    B, N, D_in = input1.shape
    _, _, D_out = mat2.shape

    # Allocate memory for intermediate results
    if inplace:
        Y_ptr = input1.data_ptr()
    else:
        Y_ptr = out.data_ptr() if out is not None else input1.new_empty((B, N, D_out)).data_ptr()

    # Launch the kernel
    fused_mul_add_logsoftmax_dropout_bmm_kernel[B, N](input1.data_ptr(), input2.data_ptr(), other.data_ptr(), mat2.data_ptr(), Y_ptr, None,
                                                       input1.stride(0), input2.stride(0), other.stride(0), mat2.stride(0), Y_ptr, None,
                                                       B, N, D_in, D_out, p, training, dim, inplace, None,
                                                       BLOCK_SIZE)

    if inplace:
        return input1
    else:
        return out if out is not None else input1.new_empty((B, N, D_out))
