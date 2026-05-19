import triton
import triton.language as tl
import torch

# Triton kernel for repeat interleave and log-softmax
@triton.jit
def _fused_repeat_interleave_log_softmax(X, OUT, repeats, dim, output_size, 
                                          xm_stride, xn_stride, out_stride, 
                                          N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    # Initialize output tensor
    out = tl.zeros((output_size,), tl.float32)
    
    # Repeat interleave operation
    for i in range(N):
        for r in range(repeats[i]):
            idx = i * repeats[i] + r
            out[idx] = X[i]

    # Log-softmax operation
    max_val = tl.max(out, axis=dim)
    exp_vals = tl.exp(out - max_val)
    sum_exp = tl.sum(exp_vals, axis=dim)
    log_softmax = out - max_val - tl.log(sum_exp)

    # Store the result in the output tensor
    tl.store(OUT + rm * out_stride, log_softmax)

# Wrapper function to call the Triton kernel
def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None) -> torch.Tensor:
    assert input.is_cuda
    if dim is None:
        input = input.view(-1)
        dim = 0

    N = input.shape[dim]
    output = input.new_empty(output_size, dtype=dtype) if out is None else out

    # Call the Triton kernel
    _fused_repeat_interleave_log_softmax[(N,)](input, output, repeats, dim, output_size,
                                                input.stride(0), input.stride(1), 
                                                output.stride(0), N, BLOCK_N=1024)
    return output
