import triton
import triton.language as tl

@triton.jit
def batch_norm_and_hardsigmoid_kernel(
    X_ptr, Y_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    N, C, H, W, stride_c, stride_h, stride_w,
    epsilon):
    # Get thread indices
    pid = tl.program_id(0)
    num_pid = tl.cdiv(N * C * H * W, 256)

    if pid >= num_pid:
        return

    # Linear index to 4D index
    n = pid // (C * H * W)
    c = (pid % (C * H * W)) // (H * W)
    h = ((pid % (C * H * W)) % (H * W)) // W
    w = (pid % (C * H * W)) % W

    # Load data from global memory
    x = tl.load(X_ptr + pid * stride_c, mask=pid < N * C * H * W, other=0.0)

    # Normalize
    var_inv = 1.0 / tl.maximum(tl.load(running_var_ptr + c), epsilon)
    mu = tl.load(running_mean_ptr + c)
    x_norm = (x - mu) * var_inv

    # Apply weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        x_norm = x_norm * tl.load(weight_ptr + c) + tl.load(bias_ptr + c)

    # Apply hardsigmoid
    y = 0.5 * (tl.relu(x_norm + 3.0) + tl.relu(-x_norm + 1.0))

    # Store result in global memory
    tl.store(Y_ptr + pid * stride_c, y, mask=pid < N * C * H * W)
