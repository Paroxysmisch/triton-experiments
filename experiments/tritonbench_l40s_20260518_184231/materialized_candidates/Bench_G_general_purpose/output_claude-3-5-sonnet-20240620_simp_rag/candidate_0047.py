import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd_kernel(
    X, Y, W, Rstd,
    stride_ml, stride_n,
    L, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * stride_ml + batch * stride_n
    Y += base_idx
    X += base_idx

    # Compute RMS
    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += a * a
    rms = tl.sqrt(tl.sum(_rms) / N + eps)

    # Store the reciprocal of the standard deviation
    tl.store(Rstd + row * L + batch, rms)

    # Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x / rms
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def rmsnorm_bwd_kernel(
    input_ptr, weight_ptr, grad_output_ptr,
    input_row_stride, grad_input_ptr, grad_weight_accum_ptr,
    num_elements, eps,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, block_size)

    # Calculate memory access ranges
    input_offsets = row_idx * input_row_stride + offsets
    input_ptrs = input_ptr + input_offsets
    weight_ptrs = weight_ptr + offsets
    grad_output_offsets = grad_output_ptr + input_offsets

    # Create masks for valid elements
    valid_elements_mask = offsets < num_elements

    # Load input values, weights, and gradient outputs
    input_values = tl.load(input_ptrs, mask=valid_elements_mask, other=0)
    weights = tl.load(weight_ptrs, mask=valid_elements_mask, other=0)
    grad_outputs = tl.load(grad_output_offsets, mask=valid_elements_mask, other=0)

    # Compute normalization factor
    norm_factor = tl.sqrt(tl.sum(input_values * input_values) / num_elements + eps)

    # Compute gradients w.r.t weights
    grad_weight_partial = input_values * grad_outputs / norm_factor
    tl.store(
        grad_weight_accum_ptr + input_offsets,
        grad_weight_partial,
        mask=valid_elements_mask,
    )

    # Compute gradients w.r.t input
    grad_input_first_term = grad_outputs * weights / norm_factor
    grad_input_second_term = (
        tl.sum(input_values * grad_outputs * weights)
        * input_values
        / (num_elements * norm_factor * norm_factor * norm_factor)
    )
    grad_input_values = grad_input_first_term - grad_input_second_term
    tl.store(
        grad_input_ptr + input_offsets, grad_input_values, mask=valid_elements_mask
    )

# Wrapper functions for the kernels
def rmsnorm_forward(x, weight, eps=1e-6, block_size=1024):
    M, L, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((M, L), dtype=x.dtype, device=x.device)
    
    grid = (M, L)
    rmsnorm_fwd_kernel[grid](
        x, y, weight, rstd,
        x.stride(0), x.stride(1),
        L, N, eps, block_size
    )
    return y, rstd

def rmsnorm_backward(grad_output, input, weight, rstd, eps=1e-6, block_size=1024):
    M, L, N = input.shape
    grad_input = torch.empty_like(input)
    grad_weight = torch.zeros_like(weight)
    
    grid = (M * L,)
    rmsnorm_bwd_kernel[grid](
        input, weight, grad_output,
        input.stride(0) * L, grad_input, grad_weight,
        N, eps, block_size
    )
    return grad_input, grad_weight.sum(0)
