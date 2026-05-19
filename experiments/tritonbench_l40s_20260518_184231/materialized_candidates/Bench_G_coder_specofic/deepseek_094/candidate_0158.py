import triton
import numpy as np

# Triton kernel
@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr, stride_x, stride_w, stride_output, N_SIZE, eps, BLOCK_N_SIZE):
    pid_m = triton.program_id(axis=0)
    pid_batch = triton.program_id(axis=1)

    # Initialize accumulators
    sum_sq = triton.operands(dtype=triton.float32, shape=())

    # Iterate over chunks of size BLOCK_N_SIZE
    for pid_n in range(0, N_SIZE, BLOCK_N_SIZE):
        # Load data from memory
        x = triton.load(x_ptr + pid_m * stride_x + pid_n, dtype=triton.float32)
        w = triton.load(rms_w_ptr + pid_m * stride_w + pid_n, dtype=triton.float32)

        # Accumulate sum of squares
        sum_sq += x * x

    # Calculate variance and standard deviation
    var = sum_sq / N_SIZE
    rstd = triton.sqrt(var) + eps

    # Normalize input and scale by weights
    for pid_n in range(0, N_SIZE, BLOCK_N_SIZE):
        x = triton.load(x_ptr + pid_m * stride_x + pid_n, dtype=triton.float32)
        w = triton.load(rms_w_ptr + pid_m * stride_w + pid_n, dtype=triton.float32)
        output = x / rstd * w

        # Write output back to memory
        triton.store(output_ptr + pid_m * stride_output + pid_n, output)

# Triton wrapper
def rmsnorm_triton_wrapper(x, rms_w, output, BATCH_SIZE, M, N_SIZE, eps, BLOCK_N_SIZE):
    # Initialize output tensor
    output[:] = 0

    # Get strides for input tensor, weight, and output tensor
    stride_x = np.array(x.strides)
    stride_w = np.array(rms_w.strides)
    stride_output = np.array(output.strides)

    # Launch Triton kernel
    rmsnorm_triton[BATCH_SIZE, M](x.ctypes.data, rms_w.ctypes.data, output.ctypes.data, stride_x, stride_w, stride_output, N_SIZE, eps, BLOCK_N_SIZE)
