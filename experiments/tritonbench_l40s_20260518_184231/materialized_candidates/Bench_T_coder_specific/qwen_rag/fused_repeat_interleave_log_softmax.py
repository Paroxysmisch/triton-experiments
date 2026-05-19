import triton
import triton.language as tl

@triton.jit
def fused_repeat_interleave_log_softmax_kernel(
    X, R, Y, XM_STRIDE, RM_STRIDE, RN_STRIDE, XM_SIZE, RM_SIZE, RN_SIZE, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr
):
    m = tl.program_id(0)
    r = tl.program_id(1)

    # Load data from input tensors
    xm = m * XM_STRIDE + r * RN_STRIDE
    xm_data = tl.load(X + xm, mask=r < XM_SIZE, other=0.0)

    # Calculate the repeated index
    ym = m * XM_STRIDE + r * RN_STRIDE
    yr = r % RN_SIZE
    Y[ym] = xm_data

    # Compute log-softmax
    max_val = tl.max(Y[ym], axis=0)
    exp_values = tl.exp(Y[ym] - max_val)
    sum_exp = tl.sum(exp_values, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_val

    # Store the result
    Y[ym] = log_sum_exp
