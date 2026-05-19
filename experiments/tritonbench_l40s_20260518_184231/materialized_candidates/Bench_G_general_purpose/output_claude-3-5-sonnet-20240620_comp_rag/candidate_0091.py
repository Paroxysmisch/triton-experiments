import triton
import triton.language as tl
import torch

# Constants and utility functions
MAX_FUSED_SIZE = 65536 // 4  # Maximum block size for tensor fusion

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    elif BLOCK_SIZE >= 8192:
        return 16
    elif BLOCK_SIZE >= 2048:
        return 8
    return 4

@triton.jit
def _rms_layernorm_forward(
    Y_ptr, Y_stride,
    X_ptr, X_stride, 
    W_ptr, W_stride,
    r_ptr, r_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and prepare offsets
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Compute base pointers
    Y_row_ptr = Y_ptr + row_idx * Y_stride
    X_row_ptr = X_ptr + row_idx * X_stride
    r_row_ptr = r_ptr + row_idx * r_stride
    
    # Load input and weights
    X_row = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute RMS normalization
    square_sum = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(square_sum + eps)
    
    # Store inverse variance for backward pass
    tl.store(r_row_ptr, inv_var)
    
    # Normalize and scale
    normalized = X_row * inv_var
    normalized = normalized.to(W_row.dtype)
    output = normalized * W_row
    
    # Store result
    tl.store(Y_row_ptr + col_offsets, output, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    dY_ptr, dY_stride,
    X_ptr, X_stride,
    W_ptr, W_stride,
    r_ptr, r_stride,
    dW_ptr, dW_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and prepare offsets
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Compute base pointers
    dY_row_ptr = dY_ptr + row_idx * dY_stride
    X_row_ptr = X_ptr + row_idx * X_stride
    r_row_ptr = r_ptr + row_idx * r_stride
    
    # Load inputs
    dY_row = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    X_row = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    inv_var = tl.load(r_row_ptr).to(tl.float32)
    
    # Compute normalized input
    normalized = X_row * inv_var
    
    # Compute gradient components
    dY_W = dY_row * W_row
    sum_dY_norm = tl.sum(dY_W * normalized, axis=0)
    
    # Final gradient computation
    grad = inv_var / n_cols * (n_cols * dY_W - normalized * sum_dY_norm)
    
    # Store gradients
    tl.store(dY_row_ptr + col_offsets, grad, mask=mask)

class RMSLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        # Reshape input if needed
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        n_rows, n_cols = x.shape
        
        # Calculate block size and warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(f"Input size {n_cols} exceeds maximum block size {MAX_FUSED_SIZE}")
        num_warps = get_num_warps(BLOCK_SIZE)
        
        # Allocate output tensors
        output = torch.empty_like(x)
        inv_var = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        
        # Launch kernel
        _rms_layernorm_forward[(n_rows,)](
            output, output.stride(0),
            x, x.stride(0),
            weight, weight.stride(0),
            inv_var, inv_var.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, inv_var)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        
        return output.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_var = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        # Reshape if needed
        orig_shape = grad_output.shape
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        n_rows, n_cols = grad_output.shape
        
        # Allocate gradient tensors
        grad_input = torch.empty_like(grad_output)
        
        # Launch backward kernel
        _rms_layernorm_backward[(n_rows,)](
            grad_output, grad_output.stride(0),
            x, x.stride(0),
            weight, weight.stride(0),
            inv_var, inv_var.stride(0),
            grad_input, grad_input.stride(0),
            n_cols, ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps
        )
        
        return grad_input.view(orig_shape), None, None
