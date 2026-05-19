import triton
from triton.language import *

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    x_ptr, y_ptr, z_ptr, other_ptr, eps, dropout_p, training,
    block_size: tl.constexpr, stride_x: tl.constexpr, stride_y: tl.constexpr,
    stride_z: tl.constexpr, stride_other: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(block_size, block_size)

    # Batch matrix multiplication
    n = pid // grid_size
    m = pid % grid_size
    b = 0  # Assuming batch size is 1 for simplicity
    z = tl.zeros((block_size,), dtype=tl.float32)
    for k in range(stride_x // block_size):
        x_k = tl.load(x_ptr + b * stride_x + n * block_size + k * block_size)
        y_k = tl.load(y_ptr + b * stride_y + k * block_size + m * block_size)
        z += x_k * y_k
    tl.store(z_ptr + b * stride_z + n * block_size + m * block_size, z)

    # RMS normalization
    norm = tl.zeros((block_size,), dtype=tl.float32)
    for k in range(block_size):
        norm[k] = z[k] * z[k]
    norm_sum = tl.sum(norm, axis=0)
    norm_value = tl.sqrt(norm_sum / block_size + eps)
    tl.store(z_ptr + b * stride_z + n * block_size + m * block_size, z / norm_value)

    # GELU activation
    g = tl.zeros((block_size,), dtype=tl.float32)
    for k in range(block_size):
        g[k] = 0.5 * (z[k] + 0.044715 * z[k] * z[k] * z[k])
    tl.store(z_ptr + b * stride_z + n * block_size + m * block_size, g)

    # Dropout
    if training:
        drop = tl.zeros((block_size,), dtype=tl.float32)
        for k in range(block_size):
            drop[k] = g[k] if tl.rand() > dropout_p else 0.0
        tl.store(z_ptr + b * stride_z + n * block_size + m * block_size, drop)
    else:
        tl.store(z_ptr + b * stride_z + n * block_size + m * block_size, g)

    # Subtraction
    tl.store(z_ptr + b * stride_z + n * block_size + m * block_size,
              z - tl.load(other_ptr + b * stride_other + n * block_size + m * block_size))

@triton.autotune
def fused_bmm_rmsnorm_gelu_dropout_sub(
    input1, input2, other, normalized_shape, dropout_p=0.5, training=True,
    approximate='none', eps=1e-5, out=None):
    
    block_size = 32
    stride_x = input1.shape[1] * input1.shape[2]
    stride_y = input2.shape[1] * input2.shape[2]
    stride_z = input1.shape[1] * input2.shape[2]
    stride_other = input1.shape[1] * input1.shape[2]

    fused_bmm_rmsnorm_gelu_dropout_sub_kernel[
        grid=(tl.cdiv(input1.shape[1], block_size), input1.shape[2], input1.shape[3]),
        block=(block_size, 1, 1),
        num_warps=4,
    ](input1.data_ptr(), input2.data_ptr(), input1.data_ptr(), other.data_ptr(),
      eps, dropout_p, training,
      block_size, stride_x, stride_y, stride_z, stride_other)

    return input1
