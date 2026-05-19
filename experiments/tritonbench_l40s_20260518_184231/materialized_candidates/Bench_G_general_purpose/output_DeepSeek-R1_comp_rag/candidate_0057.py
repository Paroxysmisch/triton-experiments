import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weights
    y_ptr,  # Pointer to output tensor
    stride_x_batch, stride_x_m, stride_x_k,  # Input strides
    stride_w_k,  # Weight stride
    stride_y_batch, stride_y_m, stride_y_k,  # Output strides
    N,  # Number of columns per row
    eps,  # Small epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size (power of two)
):
    # Get 2D program IDs for batch and row indices
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Calculate base offsets for input and output
    row_offset_x = pid_batch * stride_x_batch + pid_m * stride_x_m
    row_offset_y = pid_batch * stride_y_batch + pid_m * stride_y_m
    
    # Initialize sum of squares accumulator
    sum_squares = 0.0
    
    # First pass: Compute sum of squares in blocks
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # Load input data and convert to float32 for precision
        x = tl.load(x_ptr + row_offset_x + cols * stride_x_k, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        
        # Accumulate sum of squares (block-wise reduction)
        sum_squares += tl.sum(x_f32 * x_f32, axis=0)
    
    # Compute variance and RMS reciprocal (rrms)
    variance = sum_squares / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    
    # Second pass: Normalize and apply weights
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # Load input and weights
        x = tl.load(x_ptr + row_offset_x + cols * stride_x_k, mask=mask)
        w = tl.load(w_ptr + cols * stride_w_k, mask=mask)
        
        # Compute normalized values
        x_normalized = x * rrms
        y = x_normalized * w
        
        # Store results
        tl.store(y_ptr + row_offset_y + cols * stride_y_k, y, mask=mask)


class RmsNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Ensure contiguous tensor layout
        x = x.contiguous()
        weight = weight.contiguous()
        
        # Validate input dimensions
        batch, M, K = x.shape
        assert weight.shape == (K,), f"Weight shape {weight.shape} != ({K},)"
        
        # Allocate output tensor
        y = torch.empty_like(x)
        
        # Configure kernel launch parameters
        grid = (batch, M)  # 2D grid for batch and rows
        BLOCK_SIZE = 1024  # Tuned for modern GPUs
        
        # Launch RMS normalization kernel
        rms_norm_kernel[grid](
            x, weight, y,
            x.stride(0), x.stride(1), x.stride(2),
            weight.stride(0),
            y.stride(0), y.stride(1), y.stride(2),
            N=K, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=BLOCK_SIZE // 32  # 32 threads per warp
        )
        
        # Save for backward pass (omitted for brevity)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass implementation would go here
        raise NotImplementedError("Backward pass not implemented for this example")


def rms_norm(x, normalized_shape, weight, eps=1e-6):
    # Input validation
    assert all(x.size()[-len(normalized_shape):] == normalized_shape), \
        "Input shape mismatch for normalization"
    
    return RmsNormFunction.apply(x, weight, eps)
