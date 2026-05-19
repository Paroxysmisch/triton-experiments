import triton
import triton.language as tl
import torch

# Triton kernel for fused repeat interleave and log-softmax
@triton.jit
def _fused_repeat_interleave_log_softmax_kernel(
    X, Y, OUT, xm_stride, xn_stride, out_stride, N, repeats, dim, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_N
    offsets = block_start + tl.arange(0, BLOCK_N)
    mask = offsets < N

    # Load input data
    Xmn = X + offsets * xn_stride
    x = tl.load(Xmn, mask=mask, other=-float('inf'))

    # Repeat interleave
    repeated_x = tl.repeat(x, repeats, dim)

    # Compute max for numerical stability
    max_val = tl.max(repeated_x, axis=0)
    repeated_x -= max_val

    # Compute exponentials
    exp_x = tl.exp(repeated_x)

    # Compute sum of exponentials
    sum_exp = tl.sum(exp_x, axis=0)

    # Compute log-softmax
    log_softmax = repeated_x - tl.log(sum_exp)

    # Store the result
    Ymn = Y + offsets * out_stride
    tl.store(Ymn, log_softmax, mask=mask)

# Wrapper function to call the Triton kernel
def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    assert input.is_cuda, "Input tensor must be on a CUDA device."
    
    if dim is None:
        input = input.flatten()
        dim = 0

    # Compute the size of the repeated tensor
    if output_size is None:
        output_size = input.size(dim) * repeats

    # Ensure the output tensor is created if not provided
    if out is None:
        out = torch.empty_like(input, dtype=dtype, device=input.device)
        out = out.expand_as(input).repeat_interleave(repeats, dim=dim)
    else:
        assert out.shape == input.shape[:dim] + (output_size,) + input.shape[dim+1:], "Output tensor shape mismatch."

    # Compute the strides
    input_stride = input.stride()
    out_stride = out.stride()

    # Launch the Triton kernel
    grid = (out.numel() // output_size, )
    _fused_repeat_interleave_log_softmax_kernel[grid](
        input, input, out, input_stride[dim], input_stride[0], out_stride[dim], input.size(dim), repeats, dim, BLOCK_N=1024
    )

    return out

# Example usage
input_tensor = torch.randn(2, 3, device='cuda')
repeats = 2
dim = 1
output_tensor = fused_repeat_interleave_log_softmax(input_tensor, repeats, dim)
print(output_tensor)
