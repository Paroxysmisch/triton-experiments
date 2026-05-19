import triton
import triton.language as tl

@triton.jit
def kernel_example(X_ptr, W_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_weight_k, stride_weight_n,
                   stride_output_batch, stride_output_m, stride_output_n,
                   N_SIZE: tl.constexpr, eps: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, N_SIZE)

    # Example logic

def kernel_example_wrapper(X, W, eps=1e-6):
    batch, M, K = X.shape
    N = W.shape[1]
    assert W.shape[0] == K
    output = torch.empty((batch, M, N), dtype=W.dtype, device=W.device)
    kernel_example[(batch, M,)](X, W, output,
                                *X.stride(),
                                *W.stride(),
                                *output.stride(),
                                N_SIZE=N, eps=eps)
    return output
